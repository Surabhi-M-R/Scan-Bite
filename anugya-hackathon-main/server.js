require('dotenv').config();
const express = require('express');
const multer = require('multer');
const Tesseract = require('tesseract.js');
const sharp = require('sharp');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const path = require('path');
const fs = require('fs');

const app = express();

// 1. GEMINI AI INIT
if (!process.env.GEMINI_API_KEY) {
  console.error('Error: GEMINI_API_KEY is not set in environment variables');
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ 
  model: "gemini-1.5-flash",
  apiVersion: "v1"
});

// 2. APP CONFIGURATION
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Create uploads directory if it doesn't exist
const uploadDir = path.join(__dirname, 'public', 'uploads');
if (!fs.existsSync(uploadDir)) {
  try {
    fs.mkdirSync(uploadDir, { recursive: true });
  } catch (err) {
    console.error('Error creating uploads directory:', err);
    process.exit(1);
  }
}

// 3. FILE UPLOAD SETUP
const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, uploadDir),
    filename: (req, file, cb) => cb(null, Date.now() + path.extname(file.originalname))
  }),
  fileFilter: (req, file, cb) => {
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.mimetype)) {
      return cb(new Error('Invalid file type. Only JPEG, PNG, and WebP images are allowed.'));
    }
    cb(null, true);
  },
  limits: { 
    fileSize: 5 * 1024 * 1024,
    files: 1
  }
});

// 4. IMAGE PROCESSING
async function preprocessImage(imagePath) {
  try {
    console.log('Starting image preprocessing...');
    const processedPath = path.join(uploadDir, 'processed_' + path.basename(imagePath));
    
    await sharp(imagePath)
      .greyscale()
      .normalize()
      .linear(1.1)
      .sharpen()
      .threshold(128) // Add threshold for better text contrast
      .modulate({
        brightness: 1.2, // Increase brightness
        saturation: 0 // Remove color
      })
      .toFile(processedPath);

    console.log('Image preprocessing completed');
    return processedPath;
  } catch (err) {
    console.error('Image preprocessing error:', err);
    throw new Error('Failed to process image. Please try again with a different image.');
  }
}

async function extractText(imagePath) {
  try {
    console.log('Starting text extraction...');
    const { data: { text } } = await Tesseract.recognize(
      imagePath,
      'eng',
      { 
        logger: m => {
          if (m.status === 'recognizing text') {
            console.log('Progress:', m.progress);
          } else {
            console.log(m.status);
          }
        },
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:()[]{}@#$%^&*!?-_=+<>/\\|"\' ',
        tessedit_pageseg_mode: '6' // Assume uniform text block
      }
    );

    console.log('Text extraction completed');
    
    if (!text || text.trim().length === 0) {
      throw new Error('No text could be extracted from the image. Please try again with a clearer image.');
    }

    // Clean up the extracted text
    const cleanedText = text
      .replace(/\s+/g, ' ')
      .trim();

    console.log('Extracted text sample:', cleanedText.substring(0, 100) + '...');
    return cleanedText;
  } catch (err) {
    console.error('Text extraction error:', err);
    throw new Error('Failed to extract text from image. Please try again with a clearer image.');
  }
}

// 5. GEMINI ANALYSIS
async function analyzeIngredients(ingredients) {
  try {
    if (!ingredients || ingredients.trim().length === 0) {
      throw new Error('No ingredients text provided for analysis');
    }

    const prompt = `Analyze these food ingredients and provide a safety assessment. Ingredients: ${ingredients.substring(0, 5000)}

    Provide the analysis in this exact JSON format:
    {
      "productName": "string",
      "keyIngredients": ["list of main ingredients"],
      "safetyAssessment": {
        "isSafe": boolean,
        "reason": "string",
        "warnings": ["list of warnings if any"]
      },
      "safetyRating": {
        "score": number,
        "explanation": "string"
      },
      "harmfulChemicals": ["list of harmful chemicals if any"],
      "healthBenefits": ["list of health benefits if any"],
      "consumptionRecommendations": "string",
      "alternativeProducts": ["list of safer alternatives if any"]
    }

    Focus on safety and health aspects. If any information is not available, use appropriate default values.`;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();
    
    // Clean the response text
    const cleanText = text.replace(/```json|```/g, '').trim();
    
    try {
      const analysis = JSON.parse(cleanText);
      
      // Validate required fields
      if (!analysis.productName || !analysis.safetyAssessment || !analysis.safetyRating) {
        throw new Error('Invalid analysis response format');
      }
      
      return analysis;
    } catch (parseError) {
      console.error('JSON Parse Error:', parseError);
      console.error('Raw response:', cleanText);
      throw new Error('Failed to parse analysis response');
    }
  } catch (err) {
    console.error('Gemini Analysis Error:', err);
    throw new Error('AI analysis failed. Please try again with clearer text.');
  }
}

// 6. DETAILED ANALYSIS
async function getDetailedAnalysis(ingredients, productName) {
  try {
    if (!ingredients || !productName) {
      throw new Error('Missing required parameters for detailed analysis');
    }

    const prompt = `Analyze this food product "${productName}" with ingredients: ${ingredients.substring(0, 5000)}

    Provide a detailed analysis in the following JSON format:
    {
      "nutritionalInfo": {
        "calories": "string",
        "macronutrients": {
          "protein": "string",
          "carbs": "string",
          "fats": "string"
        },
        "vitamins": ["list"],
        "minerals": ["list"]
      },
      "healthImpact": {
        "shortTerm": "string",
        "longTerm": "string",
        "risks": ["list"]
      },
      "dietaryConsiderations": {
        "suitableFor": ["list"],
        "restrictions": ["list"],
        "allergies": ["list"]
      },
      "storage": {
        "conditions": "string",
        "shelfLife": "string",
        "precautions": ["list"]
      },
      "environmentalImpact": {
        "packaging": "string",
        "sustainability": "string",
        "ecoFriendliness": "string"
      }
    }

    Focus on safety and health aspects. If any information is not available, use "Not specified" as the value.`;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();
    
    // Clean the response text
    const cleanText = text.replace(/```json|```/g, '').trim();
    
    try {
      const analysis = JSON.parse(cleanText);
      
      // Validate the structure
      if (!analysis.nutritionalInfo || !analysis.healthImpact) {
        throw new Error('Invalid response structure');
      }
      
      return analysis;
    } catch (parseError) {
      console.error('JSON Parse Error:', parseError);
      console.error('Raw response:', cleanText);
      throw new Error('Failed to parse analysis response');
    }
  } catch (err) {
    console.error('Detailed Analysis Error:', err);
    throw new Error('Detailed analysis failed. Please try again.');
  }
}

// 7. ROUTES
app.get('/', (req, res) => {
  res.render('homepage', {
    title: 'ANUGYA',
    appName: 'ANUGYA',
    tagline: 'YOUR HEALTH SCANNER',
    description: 'Building a web application to scan packaged food products to know the ingredients and suitability to consume according to user health conditions and many more features.',
    workflowSteps: [
      {
        image: '/assets/Frame.png',
        title: 'UPLOAD',
        description: 'Upload an image of a packaged food product to analyze its ingredients.'
      },
      {
        image: '/assets/IMAGE (2).png',
        title: 'GET RESULT',
        description: 'AI scans and detects the ingredients, highlighting potential health risks.'
      },
      {
        image: '/assets/IMAGE (3).png',
        title: 'PERSONAL ADVICE',
        description: 'Receive a health rating, consumption recommendations, and better alternatives.'
      }
    ],
    ctaText: 'GET STARTED'
  });
});

app.get('/upload', (req, res) => res.render('index'));

app.post('/detail', upload.single('image'), async (req, res) => {
  try {
    let imagePath;
    let filename;

    if (!req.file) {
      throw new Error('No image file received. Please try again.');
    }

    imagePath = req.file.path;
    filename = req.file.filename;

    console.log('Processing image:', imagePath);

    // Preprocess the image
    const processedPath = await preprocessImage(imagePath);
    console.log('Image preprocessed:', processedPath);

    // Extract text from the image
    const text = await extractText(processedPath);
    console.log('Text extracted:', text.substring(0, 100) + '...');

    if (!text || text.trim().length === 0) {
      throw new Error('Could not read text from the image. Please try again with a clearer image.');
    }

    // Analyze the ingredients
    const analysis = await analyzeIngredients(text);
    console.log('Basic analysis completed');

    // Get detailed analysis
    const detailedAnalysis = await getDetailedAnalysis(text, analysis.productName);
    console.log('Detailed analysis completed');

    // Cleanup processed image
    try {
      fs.unlinkSync(processedPath);
      console.log('Processed image cleaned up');
    } catch (err) {
      console.error('Error cleaning up processed image:', err);
    }

    // Render the results
    res.render('detail', {
      imagePath: `/uploads/${filename}`,
      analysis: analysis,
      detailedAnalysis: detailedAnalysis
    });
    
  } catch (err) {
    console.error('Upload Error:', err);
    
    // Clean up any uploaded files in case of error
    if (req.file) {
      try {
        fs.unlinkSync(req.file.path);
      } catch (cleanupErr) {
        console.error('Error cleaning up uploaded file:', cleanupErr);
      }
    }

    res.status(500).render('error', { 
      message: err.message || 'An error occurred while processing your image. Please try again.'
    });
  }
});

app.get('/final', (req, res) => {
  try {
    if (!req.query.analysis || !req.query.detailedAnalysis || !req.query.imagePath) {
      throw new Error('Missing required parameters');
    }

    const analysis = JSON.parse(decodeURIComponent(req.query.analysis));
    const detailedAnalysis = JSON.parse(decodeURIComponent(req.query.detailedAnalysis));
    
    res.render('final', {
      imagePath: req.query.imagePath,
      analysis: analysis,
      detailedAnalysis: detailedAnalysis
    });
  } catch (err) {
    console.error('Error in final route:', err);
    res.status(500).render('error', { 
      message: 'Error displaying results. Please try again.'
    });
  }
});

// Add this route before the error handler
app.get('/mood', (req, res) => {
  res.render('mood');
});

// Global error handler
app.use((err, req, res, next) => {
  console.error('Global Error:', err);
  res.status(500).render('error', {
    message: 'An unexpected error occurred. Please try again.'
  });
});

// 8. START SERVER
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log('Environment:', process.env.NODE_ENV || 'development');
});


// Mood-based food recommendations route
app.post('/get-mood-food-recommendations', async (req, res) => {
  try {
    const { mood } = req.body;
   
    if (!mood) {
      return res.status(400).json({ error: 'Mood is required' });
    }

    // Define mood categories and their variations
    const moodCategories = {
      positive: [
        'happy', 'joyful', 'excited', 'energetic', 'motivated', 'peaceful', 
        'grateful', 'optimistic', 'confident', 'inspired', 'relaxed', 'content'
      ],
      negative: [
        'sad', 'anxious', 'stressed', 'tired', 'depressed', 'angry', 
        'frustrated', 'overwhelmed', 'lonely', 'worried', 'irritable', 'exhausted'
      ],
      neutral: [
        'focused', 'calm', 'balanced', 'mindful', 'centered', 'reflective',
        'contemplative', 'curious', 'thoughtful', 'present', 'aware', 'grounded'
      ]
    };

    // Determine mood category
    let moodCategory = 'neutral';
    for (const [category, moods] of Object.entries(moodCategories)) {
      if (moods.includes(mood.toLowerCase())) {
        moodCategory = category;
        break;
      }
    }

    // Construct the prompt for Gemini API with mood-specific context
    const prompt = `Provide 10-15 healthy food recommendations that help with ${mood} mood. 
    Consider the following aspects:
    - Foods that naturally boost mood and energy
    - Foods rich in mood-regulating nutrients
    - Foods that help with stress reduction
    - Foods that promote better sleep (if relevant)
    - Foods that support mental clarity
    - Foods that help with emotional balance

    For each food, include:
    - A brief explanation of how it helps with this specific mood
    - Key nutrients that contribute to mood improvement
    - Simple preparation suggestions
    - Best time to consume

    Format the response as a JSON object with:
    {
      "moodCategory": "${moodCategory}",
      "foods": [
        {
          "name": "string",
          "benefits": "string",
          "nutrients": ["list of key nutrients"],
          "preparation": "string",
          "bestTime": "string"
        }
      ],
      "additionalTips": {
        "dietaryAdvice": "string",
        "lifestyleTips": ["list of tips"],
        "avoidFoods": ["list of foods to avoid"]
      }
    }

    Make the recommendations specific to ${mood} mood and ensure they are practical and easy to implement.`;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();
   
    // Clean the response and parse as JSON
    const cleanText = text.replace(/```json|```/g, '');
    const recommendations = JSON.parse(cleanText);
   
    res.json(recommendations);
   
  } catch (err) {
    console.error('Mood Food Recommendation Error:', err);
    res.status(500).json({ error: 'Failed to generate food recommendations' });
  }
});



