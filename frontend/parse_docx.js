const fs = require('fs');
const StreamZip = require('node-stream-zip');

async function extractText() {
    const zip = new StreamZip.async({ file: process.argv[2] });
    try {
        const xmlData = await zip.entryData('word/document.xml');
        const xmlString = xmlData.toString('utf8');
        const text = xmlString.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
        fs.writeFileSync('AuraDiff_ImplPlan_v2.1_node.txt', text);
        console.log("Success");
    } catch (e) {
        console.error(e);
    } finally {
        await zip.close();
    }
}
extractText();
