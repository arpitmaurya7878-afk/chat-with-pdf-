const express = require('express')
const multer = require('multer')
const pdfParse = require('pdf-parse')
const fs = require('fs')
const { GoogleGenAI } = require('@google/genai')
const { QdrantClient } = require('@qdrant/js-client-rest')
const cors = require('cors')

require('dotenv').config()

const app = express()
app.use(cors())
app.use(express.json())
const upload = multer({ dest: "uploads" })

const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY
})

const qdrant = new QdrantClient({
    url: process.env.QDRANT_DB_URL,
    apiKey: process.env.QDRANT_API_KEY
})


async function createEmbedding(text) {

    const response = await ai.models.embedContent({
        model: 'gemini-embedding-2',
        contents: text
    })

    return response.embeddings[0].values
}


/* HOME ROUTE */

app.get('/', (req, res) => {

    res.send("i am arpit")

})


/* CREATE COLLECTION */

app.get('/create-collection', async (req, res) => {

    try {
      
        await qdrant.createCollection('pdf-docs', {

            vectors: {
                size: 3072,
                distance: "Cosine"
            }

        })

        res.send('collection is created')

    } catch (error) {

        console.log(error)

        res.status(500).send(error)

    }

})


/* UPLOAD PDF */

app.post('/upload', upload.single('pdf'), async (req, res) => {

    try {

        console.log("PDF received")
        

        const dataBuffer = fs.readFileSync(req.file.path)

        const pdfData = await pdfParse(dataBuffer)

        const text = pdfData.text


        /* CREATE CHUNKS */

        const chunks = text
            .split(/(?<=[.!?])\s+/)
            .map(chunk => chunk.trim())
            .filter(chunk => chunk.length > 0)


        console.log("Total chunks:", chunks.length)


        /* CREATE EMBEDDINGS */

        const chunkEmbedding = []

       for (const chunk of chunks) {

    console.log("Creating embedding...");

    const embedding = await createEmbedding(chunk);

    console.log("Embedding created");

    chunkEmbedding.push({
        text: chunk,
        embedding
    });
}

        /* CREATE QDRANT POINTS */

       console.log("Creating Qdrant points...");

const points = chunkEmbedding.map((item, index) => ({
    id: index + 1,
    vector: item.embedding,
    payload: {
        text: item.text,
        filename: req.file.originalname,
        chunkIndex: index
    }
}));

console.log("Points created:", points.length);
console.log("Storing data in Qdrant...");

await qdrant.upsert('pdf-docs', {
    points
});

console.log("Data stored in Qdrant!");


        /* DELETE TEMPORARY FILE */

        fs.unlinkSync(req.file.path)


        res.send({

            message: "PDF uploaded and stored successfully",

            chunks: chunks.length

        })
        


  } catch (error) {
    console.error("UPLOAD ERROR:", error);

    res.status(500).json({
        error: error.message
    });
}


/* ASK QUESTION */

app.post('/ask', async (req, res) => {

    try {

        const question = req.body.question


        if (!question) {

            return res.status(400).send("Question is required")

        }


        /* CREATE QUESTION EMBEDDING */

        const questionEmbedding = await createEmbedding(question)


        /* SEARCH QDRANT */

        const searchResult = await qdrant.query('pdf-docs', {

            query: questionEmbedding,

            limit: 3,

            with_payload: true

        })


        console.log("QDRANT RESULT:", searchResult)


        if (
            !searchResult ||
            !searchResult.points ||
            searchResult.points.length === 0
        ) {

            return res.status(404).send("No relevant chunk found")

        }


        /* GET RELEVANT CHUNKS */

        const relevantChunks = searchResult.points
            .map(point => point.payload?.text)
            .filter(text => text)


        console.log("RELEVANT CHUNKS:", relevantChunks)


        /* CREATE CONTEXT */

        const context = relevantChunks.join("\n\n")


        /* ASK GEMINI */

        const response = await ai.models.generateContent({

            model: 'gemini-3.5-flash-lite',

            contents: `
Answer the question using only the context provided below.

Context:
${context}

Question:
${question}

If the answer is not present in the context, say:
"I could not find the answer in the PDF."
`

        })


        console.log("Gemini response received")

        console.log(response.text)


        res.send({

            answer: response.text

        })


   } catch (error) {
    console.error("ASK ERROR:", error);

    res.status(500).json({
        error: error.message
    });
}


const PORT = process.env.PORT || 5000;

app.listen(PORT, "0.0.0.0", () => {
    console.log(`server running on port ${PORT}`);
});
})








