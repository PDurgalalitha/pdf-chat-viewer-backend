const fs = require("fs");
const pdfParse = require("pdf-parse");

async function extractPages(pdfPath) {
  const dataBuffer = fs.readFileSync(pdfPath);
  const data = await pdfParse(dataBuffer, { max: 0 });
  const pages = data.text
    .split("\f")
    .map((p) => p.trim())
    .filter(Boolean);
  return pages;
}
module.exports = { extractPages };
