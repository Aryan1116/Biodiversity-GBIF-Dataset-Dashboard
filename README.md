# Biodiversity Dashboard 🌿

An interactive dashboard to explore and visualize biodiversity data from the GBIF dataset. Built with [Streamlit](https://streamlit.io/) and [Folium](https://python-visualization.github.io/folium/).

## 📊 Features

*   **Global Overview**: Visualize top species classes, kingdom distribution, and taxonomic hierarchy using Sunburst charts.
*   **Temporal Trends**: Analyze seasonality and observation trends over the years.
*   **Geography & Country Analysis**:
    *   View top countries by observation count.
    *   Explore biodiversity density heatmaps.
    *   Drill down into specific country statistics (Top Species, Genera, and Richness Indices).
*   **Species Explorer**:
    *   Search for specific species.
    *   View individual sighting locations on an interactive map.
    *   Download raw sighting data as CSV.
*   **Advanced Statistics**: Explore data distributions with Boxplots and Violin plots.

## 🚀 How to Run Locally

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Aryan1116/Biodiversity-GBIF-Dataset-Dashboard.git
    cd Biodiversity-GBIF-Dataset-Dashboard
    ```

2.  **Install dependencies**:
    ```bash
    pip install streamlit pandas numpy matplotlib seaborn plotly folium streamlit-folium
    ```

3.  **Run the app**:
    ```bash
    streamlit run app.py
    ```

## 🌐 Live Demo

*(If you have deployed your app to Streamlit Cloud, paste the link here. Otherwise, you can remove this section or leave it as a placeholder)*

[**Open Dashboard**](#) *(Link coming soon)*

## 📂 Dataset

The dashboard uses `dataset_1.csv` which contains biodiversity observations including taxonomy, location (latitude/longitude), and dates.

## 🛠️ Technologies Used

*   **Python**
*   **Streamlit** (Web App Framework)
*   **Pandas & NumPy** (Data Manipulation)
*   **Matplotlib & Seaborn** (Static Visualizations)
*   **Plotly** (Interactive Charts)
*   **Folium** (Interactive Maps)
