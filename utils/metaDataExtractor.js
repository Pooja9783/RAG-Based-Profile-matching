function extractMetadata(text) {
    const skillsList = [
        "JavaScript","Python","React","Node.js","MongoDB",
        "Express","SQL","Machine Learning","AWS","Docker"
    ];
    const skills = skillsList.filter(skill =>
        text.toLowerCase().includes(skill.toLowerCase())
    );
    const experienceMatch = text.match(/(\d+)\+?\s+years?/i);
    const educationMatch = text.match(
        /(B\.Tech|B\.E|M\.Tech|Bachelor|Master|BSc|MSc)/i
    );
    const nameMatch = text.match(/^([A-Z][a-z]+\s[A-Z][a-z]+)/);
    return {
        name: nameMatch ? nameMatch[0] : "Unknown",
        skills: skills,
        experience_years: experienceMatch ? parseInt(experienceMatch[1]) : 0,
        education: educationMatch ? educationMatch[0] : "Unknown"
    };
}
module.exports = { extractMetadata };