import * as dotenv from "dotenv";
import * as fs from "fs";
dotenv.config();

import { OpenAI } from "openai";
const openai = new OpenAI(process.env.OPENAI_API_KEY); // Initialize OpenAI client with API key

async function describeImage(imageUrl) {
    try {
        const response = await openai.chat.completions.create({
            model: "gpt-4-turbo",
            messages: [
                {
                    role: "user",
                    content: [
                        { type: "text", text: "Analyze the provided image and deliver a detailed description of all identifiable objects. For each object, provide the coordinates in the format (x, y) with the origin (0, 0) located at the top-left corner of the image. Additionally, suggest which object appears to be the most important based on the context within the image along with its cordinate." },
                        {
                            type: "image_url",
                            image_url: {
                                url: imageUrl,
                                detail: "high"
                            },
                        },
                    ],
                }
            ],
            max_tokens: 500,
        });

        const description = response.choices[0].message.content;
        console.log(description);

        const output = { description, imageUrl };

        // Write the description and image URL to a file
        fs.writeFileSync("image_description.json", JSON.stringify(output), "utf8");
    } catch (error) {
        console.error("Error describing image:", error);
    }
}

// Call the describeImage function with the URL of the image
const imageUrl = "https://media.istockphoto.com/id/1369521370/photo/portrait-of-a-seller-at-a-street-market.jpg?s=612x612&w=0&k=20&c=wxyClhozOS4z-GrGbm5llbpq2TM8tD5y_j6l9r63EvE=";
describeImage(imageUrl);
