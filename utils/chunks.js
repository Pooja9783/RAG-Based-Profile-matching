const fs = require("fs");
const path = require("path");
const pdfjs = require("pdfjs-dist/legacy/build/pdf.mjs");

const resumeFolder = "./resumes";

async function loadResumes() {
    const files = fs.readdirSync(resumeFolder).filter(f => f.endsWith(".pdf"));
    let results = [];

    for (const file of files) {
        const filePath = path.join(resumeFolder, file);
        const data = new Uint8Array(fs.readFileSync(filePath));
        const pdf = await pdfjs.getDocument({ data }).promise;

        let text = "";

        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const content = await page.getTextContent();
            const pageText = content.items.map(item => item.str).join("");
            text += pageText + " ";
        }

        const cleanedText = cleanText(text);
        const chunks = chunkResumeBySentences(cleanedText);

        results.push({
            file,
            text: cleanedText,
            chunks
        });
    }

    return results;
}

function cleanText(text) {
    return text
        .replace(/\s+/g, " ")
        .replace(/-\s+/g, "")
        .replace(/\.([A-Z])/g, ". $1")
        .replace(/\s+([.,:;])/g, "$1")
        .replace(/\S+@\S+\.\S+/g, "")
        .replace(/\+?\d[\d\s-]{8,}/g, "")
        .replace(/https?:\/\/\S+/g, "")
        .trim();
}

function chunkResumeBySentences(text) {
    const maxChunkSize = 400;
    const overlap = 80;

    const sentenceRegex = /[^.!?]+[.!?]+[\])'"`’”]*|.+$/g;
    const sentences = text.match(sentenceRegex) || [];

    let chunks = [];
    let currentChunk = "";

    for (const sentence of sentences) {
        if ((currentChunk + sentence).length > maxChunkSize) {
            chunks.push(currentChunk.trim());
            const overlapText = currentChunk.slice(-overlap);
            currentChunk = overlapText + sentence;
        } else {
            currentChunk += sentence;
        }
    }

    if (currentChunk.trim()) chunks.push(currentChunk.trim());

    return chunks;
}

module.exports = { loadResumes };