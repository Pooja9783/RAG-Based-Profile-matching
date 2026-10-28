const fs = require("fs");
const { pipeline } = require("@xenova/transformers");
const { loadResumes } = require("./utils/chunks");
const { extractMetadata } = require("./utils/metaDataExtractor");

// ─────────────────────────────────────────────
// FIX 1: Expanded skill list — covers most JDs
// Old code had only 8 skills hardcoded.
// Now we have 50+ skills across multiple domains.
// ─────────────────────────────────────────────
const KNOWN_SKILLS = [
    // Frontend
    "JavaScript", "TypeScript", "React", "Vue", "Angular", "HTML", "CSS",
    "Tailwind", "Next.js", "Redux",
    // Backend
    "Node.js", "Express", "Django", "Flask", "FastAPI", "Spring Boot",
    "REST API", "GraphQL",
    // Databases
    "MongoDB", "PostgreSQL", "MySQL", "SQL", "Redis", "Firebase",
    // Cloud & DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Terraform",
    // Data / ML
    "Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
    "Pandas", "NumPy", "scikit-learn", "NLP", "LLM", "RAG",
    // General
    "Git", "Linux", "Agile", "Scrum", "Microservices", "System Design"
];

// ─────────────────────────────────────────────
// FIX 1: Dynamic skill extraction from JD text
// Old code filtered a fixed 8-skill list against the JD.
// Now we scan the JD against 50+ skills so nothing gets missed.
// ─────────────────────────────────────────────
function extractSkills(text) {
    return KNOWN_SKILLS.filter(skill =>
        text.toLowerCase().includes(skill.toLowerCase())
    );
}

// ─────────────────────────────────────────────
// FIX 1: Extract experience requirements from JD text
// Old code had no experience parsing at all.
// e.g. "5+ years Python" -> { years: 5, skill: "Python" }
// Also skips generic words like "experience", "of", "in"
// ─────────────────────────────────────────────
function extractExperienceRequirements(text) {
    const requirements = [];
    const pattern = /(\d+)\+?\s*years?\s*(?:of\s+)?([A-Za-z.+#]+)/gi;
    let match;
    while ((match = pattern.exec(text)) !== null) {
        const skill = match[2].replace(/\.$/, "");
        if (!["experience", "of", "in", "with"].includes(skill.toLowerCase())) {
            requirements.push({ years: parseInt(match[1]), skill });
        }
    }
    return requirements;
}

// ─────────────────────────────────────────────
// Cosine similarity — unchanged from your original
// ─────────────────────────────────────────────
function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function skillMatchScore(jobSkills, candidateSkills) {
    if (jobSkills.length === 0) return 0;
    const matched = jobSkills.filter(skill =>
        candidateSkills.map(s => s.toLowerCase()).includes(skill.toLowerCase())
    );
    return matched.length / jobSkills.length;
}

// ─────────────────────────────────────────────
// FIX 2: Real per-candidate reasoning
// Old code: same hardcoded string for every single candidate.
// Now: unique explanation per candidate — skills matched/missing,
//      experience check, and semantic alignment percentage.
// ─────────────────────────────────────────────
function buildReasoning(jobSkills, candidate, expRequirements) {
    const meta = candidate.metadata;
    const reasons = [];

    const matchedSkills = jobSkills.filter(skill =>
        (meta.skills || []).map(s => s.toLowerCase()).includes(skill.toLowerCase())
    );
    const missingSkills = jobSkills.filter(skill =>
        !(meta.skills || []).map(s => s.toLowerCase()).includes(skill.toLowerCase())
    );

    if (matchedSkills.length > 0) {
        reasons.push(`Matched ${matchedSkills.length}/${jobSkills.length} required skills: ${matchedSkills.join(", ")}.`);
    }
    if (missingSkills.length > 0) {
        reasons.push(`Missing skills: ${missingSkills.join(", ")}.`);
    }

    if (meta.experience_years && meta.experience_years > 0) {
        reasons.push(`Candidate has ${meta.experience_years} years of experience.`);
        expRequirements.forEach(req => {
            if (meta.experience_years >= req.years) {
                reasons.push(`Meets the ${req.years}+ years requirement for ${req.skill}.`);
            } else {
                reasons.push(`Does not meet the ${req.years}+ years requirement for ${req.skill} (has ${meta.experience_years} yrs).`);
            }
        });
    }

    const semPct = Math.round(candidate.semanticScore * 100);
    if (semPct >= 70) {
        reasons.push(`Strong semantic alignment with job description (${semPct}%).`);
    } else if (semPct >= 50) {
        reasons.push(`Moderate semantic alignment with job description (${semPct}%).`);
    } else {
        reasons.push(`Low semantic alignment with job description (${semPct}%).`);
    }

    return reasons.join(" ");
}

// ─────────────────────────────────────────────

// candidates missing any must-have skill are excluded entirely
//      before scoring and ranking happens.
// ─────────────────────────────────────────────
function passesMustHaveFilter(candidate, mustHaveSkills, expRequirements) {
    const meta = candidate.metadata;
    const candidateSkillsLower = (meta.skills || []).map(s => s.toLowerCase());

    for (const skill of mustHaveSkills) {
        if (!candidateSkillsLower.includes(skill.toLowerCase())) {
            return { passes: false, reason: `Missing must-have skill: ${skill}` };
        }
    }

    for (const req of expRequirements) {
        if (meta.experience_years !== undefined && meta.experience_years < req.years) {
            return {
                passes: false,
                reason: `Requires ${req.years}+ years for ${req.skill}, candidate has ${meta.experience_years}`
            };
        }
    }

    return { passes: true };
}

// ─────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────
async function matchJobs(jobDescription, mustHaveSkills = [], topK = 10) {

    console.log("Loading embedding model...");
    const embedder = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2"
    );

    //: Dynamically extract skills and experience requirements from JD
    const jobSkills = extractSkills(jobDescription);
    const expRequirements = extractExperienceRequirements(jobDescription);

    console.log("Detected JD skills     :", jobSkills);
    console.log("Experience requirements:", expRequirements);

    const vectorDB = JSON.parse(fs.readFileSync("vectorDB.json"));

    // Embed the job description
    const outputs = await embedder(jobDescription, { pooling: "mean", normalize: true });
    const jobEmbedding = Array.from(outputs.data);

    // Score each candidate — keep only the best-scoring chunk per candidate
    const candidateScores = {};

    vectorDB.forEach(entry => {
        const semanticScore = cosineSimilarity(jobEmbedding, entry.embedding);
        const skillScore = skillMatchScore(jobSkills, entry.metadata.skills || []);
        const finalScore = (0.7 * semanticScore) + (0.3 * skillScore);
        const file = entry.file;

        if (!candidateScores[file] || finalScore > candidateScores[file].score) {
            candidateScores[file] = { ...entry, score: finalScore, semanticScore, skillScore };
        }
    });

    const scored = Object.values(candidateScores);

    //  Filter out candidates who don't meet must-have requirements
    const filtered = [];
    const rejected = [];

    scored.forEach(candidate => {
        const result = passesMustHaveFilter(candidate, mustHaveSkills, expRequirements);
        if (result.passes) {
            filtered.push(candidate);
        } else {
            rejected.push({ file: candidate.file, reason: result.reason });
        }
    });

    if (rejected.length > 0) {
        console.log(`\nFiltered out ${rejected.length} candidates:`);
        rejected.forEach(r => console.log(`  x ${r.file}: ${r.reason}`));
    }

    // Sort and take top K
    filtered.sort((a, b) => b.score - a.score);
    const top = filtered.slice(0, topK);

    const output = {
        job_description: jobDescription,
        must_have_skills: mustHaveSkills,
        total_candidates_evaluated: scored.length,
        candidates_after_filter: filtered.length,
        top_matches: top.map(r => {
            const matchedSkills = jobSkills.filter(skill =>
                (r.metadata.skills || []).map(s => s.toLowerCase()).includes(skill.toLowerCase())
            );
            return {
                candidate_name: r.metadata.name || r.file.replace(".pdf", ""),
                resume_path: `resumes/${r.file}`,
                match_score: Math.round(r.score * 100),
                semantic_score: parseFloat(r.semanticScore.toFixed(2)),
                skill_match_score: parseFloat(r.skillScore.toFixed(2)),
                experience_years: r.metadata.experience_years || "Unknown",
                matched_skills: matchedSkills,
                all_candidate_skills: r.metadata.skills || [],
                relevant_excerpts: [r.chunk],
                // FIX 2: unique reasoning per candidate
                reasoning: buildReasoning(jobSkills, r, expRequirements)
            };
        })
    };

    console.log("\n" + JSON.stringify(output, null, 2));
    fs.writeFileSync("match_results.json", JSON.stringify(output, null, 2));
    console.log("\nResults saved to match_results.json");

    return output;
}

// ─────────────────────────────────────────────
// Run — edit jobDescription and mustHaveSkills below
// ─────────────────────────────────────────────
const jobDescription = `
    Looking for a Senior React and Node.js developer with 3+ years of experience.
    Must have strong knowledge of MongoDB and REST API design.
    Experience with Docker and AWS is a plus.
`;

// FIX 3: Candidates missing ANY of these skills are excluded from results
const mustHaveSkills = ["React", "Node.js"];

matchJobs(jobDescription, mustHaveSkills, 10);