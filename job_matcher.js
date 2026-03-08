const fs = require("fs");
const { pipeline } = require("@xenova/transformers");

function cosineSimilarity(a, b) {
    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function main() {

    const jobDescription =
        "Looking for a React Node developer with 3 years experience"

    const jobSkills = extractSkills(jobDescription);

    function skillMatchScore(jobSkills, candidateSkills) {

        const matched = jobSkills.filter(skill =>
            candidateSkills.includes(skill)
        );

        return matched.length / jobSkills.length;
    }



    const vectorDB = JSON.parse(fs.readFileSync("vectorDB.json"));

    const embedder = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2"
    );

    const outputs = await embedder(jobDescription, {
        pooling: "mean",
        normalize: true
    });

    const jobEmbedding = Array.from(outputs.data);



    const candidateScores = {};

    vectorDB.forEach(entry => {

        const semanticScore = cosineSimilarity(jobEmbedding, entry.embedding);

        const skillScore = skillMatchScore(jobSkills, entry.metadata.skills);

        const finalScore = (0.7 * semanticScore) + (0.3 * skillScore);

        const file = entry.file;

        if (!candidateScores[file] || finalScore > candidateScores[file].score) {

            candidateScores[file] = {
                ...entry,
                score: finalScore,
                semanticScore,
                skillScore
            };
        }

    });

    const scored = Object.values(candidateScores);

    scored.sort((a, b) => b.score - a.score);

    const top = scored.slice(0, 10);

    const output = {
        job_description: jobDescription,
        top_matches: top.map(r => ({
            candidate_name: r.file.replace(".pdf", ""),
            resume_path: `resumes/${r.file}`,
            match_score: Math.round(r.score * 100),
            semantic_score: r.semanticScore.toFixed(2),
            skill_match_score: r.skillScore.toFixed(2),
            matched_skills: r.metadata.skills,
            relevant_excerpts: [r.chunk],
            reasoning: "Semantic similarity between job description and resume"
        }))
    };

    console.log(JSON.stringify(output, null, 2));
}


function extractSkills(text) {
    const skills = [
        "JavaScript",
        "React",
        "Node.js",
        "MongoDB",
        "Express",
        "Python",
        "Docker",
        "SQL"
    ];

    return skills.filter(skill =>
        text.toLowerCase().includes(skill.toLowerCase())
    );
}

main();