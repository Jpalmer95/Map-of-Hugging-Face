# Map of Hugging Face 🤗

An interactive visualization of the Hugging Face ecosystem, inspired by the [Map of GitHub](https://anvaka.github.io/map-of-github/).

## 🌍 [**View Live Map**](https://jpalmer95.github.io/Map-of-Hugging-Face/)

## Features

- **🔺 Models** (5,000 items) - Triangular points
- **🔲 Datasets** (2,000 items) - Square points  
- **🔵 Spaces** (1,000 items) - Circular points

### Interactive Features
- 🔍 **Search** by name or tags
- 🎛️ **Filter** by category and territory
- 📏 **Zoom & Pan** to explore different regions
- 🖱️ **Click** any point to visit the Hugging Face page
- 💡 **Hover** for detailed tooltips

### Territories
The map organizes items into thematic territories based on their purpose:

**Models Continent:**
- NLP Core, Vision, Audio, Generative AI, Embeddings, Multimodal, etc.

**Datasets Continent:**  
- Text Corpora, Classification Data, Vision Data, Audio Data, etc.

**Spaces Continent:**
- Demos, Tools, Leaderboards, Games, etc.

## Technical Details

- **Data Source:** Hugging Face Hub API
- **Visualization:** D3.js with UMAP clustering
- **Total Items:** 8,000
- **Last Updated:** 2025-06-23

## How It Works

1. **Data Collection:** Scraped top models, datasets, and spaces using the HF Hub API
2. **Feature Extraction:** Analyzed tags and metadata for clustering
3. **Clustering:** Used UMAP for 2D projection and territory assignment
4. **Visualization:** Interactive D3.js map with zoom, search, and filtering

## Inspiration

Inspired by [Andrei Kashcha's Map of GitHub](https://github.com/anvaka/map-of-github), this project applies similar visualization techniques to the Hugging Face ecosystem.

---
Built with ❤️ using the Hugging Face Hub API and D3.js
