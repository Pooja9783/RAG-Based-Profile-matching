const fs = require("fs");
const { pipeline } = require("@xenova/transformers");
const { loadResumes } = require("./utils/chunks");
const { extractMetadata } = require("./utils/metaDataExtractor");

async function main() {

    console.log("Loading embedding model...");

    const embedder = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2"
    );

    const resumes = await loadResumes();

    const vectorDB = [];

    for (const resume of resumes) {

        const metadata = extractMetadata(resume.text);

        for (const chunk of resume.chunks) {

            const output = await embedder(chunk, { pooling: "mean", normalize: true });
            const embedding = Array.from(output.data);
            
            vectorDB.push({
                file: resume.file,
                chunk: chunk,
                embedding: embedding,
                metadata: metadata
            });

        }
    }

    fs.writeFileSync(
        "vectorDB.json",
        JSON.stringify(vectorDB, null, 2)
    );

    console.log("Vector database created with", vectorDB.length, "chunks");
}

main();