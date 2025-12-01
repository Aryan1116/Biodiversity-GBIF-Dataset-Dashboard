import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, HeatMap

# Set page configuration
st.set_page_config(page_title="Biodiversity Dashboard", layout="wide")

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset_1.csv')
        
        # Data Cleaning (from EDA)
        unuseful_col = {'elevation', 'locality', 'recordNumber', 'depthAccuracy', 'depth', 
                        'elevationAccuracy', 'coordinatePrecision', 'establishmentMeans', 
                        'typeStatus', 'verbatimScientificNameAuthorship', 'individualCount', 
                        'infraspecificEpithet', 'mediaType'}
        # Drop columns if they exist
        cols_to_drop = [c for c in unuseful_col if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
        
        # Drop rows with missing class or coordinates
        df.dropna(subset=['class', 'decimalLatitude', 'decimalLongitude'], inplace=True)
        
        # Fill missing taxonomy
        tax_cols = ['phylum','family','genus','order','species','speciesKey']
        for col in tax_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
                
        return df
    except FileNotFoundError:
        st.error("dataset_1.csv not found. Please make sure the file is in the same directory.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Country Filter
all_countries = sorted(df['countryCode'].dropna().unique())
selected_countries = st.sidebar.multiselect("Select Country", all_countries, default=all_countries[:5] if len(all_countries) > 5 else all_countries)

# Year Filter
min_year = int(df['year'].min())
max_year = int(df['year'].max())
selected_year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (2000, max_year))

# Kingdom Filter
all_kingdoms = sorted(df['kingdom'].dropna().unique())
selected_kingdoms = st.sidebar.multiselect("Select Kingdom", all_kingdoms, default=all_kingdoms)

# Filter Data
filtered_df = df[
    (df['countryCode'].isin(selected_countries)) &
    (df['year'].between(selected_year_range[0], selected_year_range[1])) &
    (df['kingdom'].isin(selected_kingdoms))
]

st.sidebar.markdown(f"**Total Observations:** {len(filtered_df)}")

# --- Main Layout ---
st.title("Biodiversity Interactive Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Temporal Trends", "Geography & Country Analysis", "Species Explorer"])

# --- Tab 1: Overview ---
with tab1:
    st.header("Global Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 15 Most Common Classes")
        if not filtered_df.empty:
            top_classes = filtered_df['class'].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(10, 5))
            top_classes.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.info("No data available for selected filters.")

    with col2:
        st.subheader("Kingdom Distribution")
        if not filtered_df.empty:
            kingdom_counts = filtered_df['kingdom'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 5))
            kingdom_counts.plot(kind='bar', ax=ax, color='lightgreen')
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.info("No data available.")

    st.subheader("Taxonomic Diversity (Sunburst)")
    if not filtered_df.empty:
        # Limit to top 1000 rows for performance if needed, or aggregate
        # Aggregating for sunburst to avoid performance issues with huge data
        sunburst_data = filtered_df.groupby(['kingdom', 'phylum', 'class', 'order']).size().reset_index(name='count')
        # Filter out small counts to keep chart readable
        sunburst_data = sunburst_data[sunburst_data['count'] > 0] 
        
        fig = px.sunburst(sunburst_data, path=['kingdom', 'phylum', 'class', 'order'], values='count', 
                          title="Taxonomic Hierarchy")
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 5: Data Statistics (Missing EDA Plots) ---
with st.expander("Advanced Data Statistics (Boxplots & Violin Plots)"):
    st.header("Data Distribution Statistics")
    
    if not filtered_df.empty:
        top_10_classes = filtered_df['class'].value_counts().head(10).index
        df_top = filtered_df[filtered_df['class'].isin(top_10_classes)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Latitude Distribution (Boxplot)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_top, x='class', y='decimalLatitude', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            st.subheader("Longitude Distribution (Boxplot)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_top, x='class', y='decimalLongitude', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with col2:
            st.subheader("Latitude Distribution (Violin Plot)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=df_top, x='class', y='decimalLatitude', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            if 'coordinateUncertaintyInMeters' in df_top.columns:
                st.subheader("Coordinate Uncertainty (Log Scale)")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=df_top, x='class', y='coordinateUncertaintyInMeters', ax=ax)
                ax.set_yscale("log")
                plt.xticks(rotation=45)
                st.pyplot(fig)
    else:
        st.info("No data available for statistics.")

# --- Tab 2: Temporal Trends ---
with tab2:
    st.header("Temporal Trends")
    
    if not filtered_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Observations per Month (Seasonality)")
            # Get top 10 classes for cleaner plot
            top_10_classes = filtered_df['class'].value_counts().head(10).index
            df_top_classes = filtered_df[filtered_df['class'].isin(top_10_classes)]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(data=df_top_classes, x='month', hue='class', ax=ax)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
            
        with col2:
            st.subheader("Observations Trend Over Years")
            year_counts = filtered_df.groupby('year').size()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(x=year_counts.index, y=year_counts.values, ax=ax)
            ax.set_ylabel("Number of Observations")
            st.pyplot(fig)
    else:
        st.info("No data available.")

# --- Tab 3: Geography & Country Analysis ---
with tab3:
    st.header("Geography & Country Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Countries by Observation Count")
        if not filtered_df.empty:
            top_countries = filtered_df['countryCode'].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(10, 5))
            top_countries.plot(kind='bar', ax=ax, color='salmon')
            ax.set_ylabel("Count")
            st.pyplot(fig)
            
    with col2:
        st.subheader("Biodiversity Density Heatmap")
        if not filtered_df.empty:
            # Using Folium for Heatmap
            map_center = [filtered_df['decimalLatitude'].mean(), filtered_df['decimalLongitude'].mean()]
            m = folium.Map(location=map_center, zoom_start=2)
            
            heat_data = [[row['decimalLatitude'], row['decimalLongitude']] for index, row in filtered_df.iterrows()]
            # Limit points for performance if too many
            if len(heat_data) > 5000:
                st.warning("Heatmap downsampled to 5000 points for performance.")
                heat_data = heat_data[:5000]
                
            HeatMap(heat_data).add_to(m)
            st_folium(m, width=700, height=500)
            
    st.divider()
    st.subheader("Specific Country Analysis")
    
    # Dropdown to select a specific country for detailed analysis
    analysis_country = st.selectbox("Select a Country for Detailed Analysis", all_countries)
    
    if analysis_country:
        country_df = df[df['countryCode'] == analysis_country]
        
        if not country_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Top 10 Species in {analysis_country}**")
                top_species = country_df['species'].value_counts().head(10)
                st.bar_chart(top_species)
                
            with c2:
                st.markdown(f"**Top 10 Genera in {analysis_country}**")
                top_genera = country_df['genus'].value_counts().head(10)
                st.bar_chart(top_genera)
                
            # Richness Indices
            st.markdown("### Biodiversity Richness (Shannon & Simpson)")
            
            # Calculate richness by state if stateProvince exists
            if 'stateProvince' in country_df.columns:
                state_richness = {}
                for state in country_df['stateProvince'].dropna().unique():
                    species_state = country_df[country_df['stateProvince'] == state]['species'].value_counts()
                    proportions = species_state / species_state.sum()
                    if len(proportions) > 0:
                        shannon = -np.sum(proportions * np.log(proportions))
                        simpson = 1 - np.sum(proportions**2)
                        state_richness[state] = [shannon, simpson]
                
                if state_richness:
                    richness_df = pd.DataFrame(state_richness, index=['Shannon', 'Simpson']).T.sort_values(by='Shannon', ascending=False)
                    st.dataframe(richness_df.head(10))
                    
                    st.markdown("**Top 10 States by Shannon Index**")
                    st.bar_chart(richness_df['Shannon'].head(10))

# --- Tab 4: Species Explorer ---
with tab4:
    st.header("Species Explorer")
    
    species_list = sorted(df['species'].dropna().unique())
    selected_species = st.selectbox("Search for a Species", species_list)
    
    if selected_species:
        species_df = df[df['species'] == selected_species]
        
        st.markdown(f"### Locations of *{selected_species}*")
        st.markdown(f"Total Sightings: {len(species_df)}")
        
        if not species_df.empty:
            # Folium Map with Markers
            m2 = folium.Map(location=[species_df['decimalLatitude'].mean(), species_df['decimalLongitude'].mean()], zoom_start=4)
            marker_cluster = MarkerCluster().add_to(m2)
            
            for idx, row in species_df.iterrows():
                folium.Marker(
                    location=[row['decimalLatitude'], row['decimalLongitude']],
                    popup=f"{row['species']} ({row['year']})",
                    tooltip=row['locality'] if 'locality' in row and pd.notna(row['locality']) else "Location"
                ).add_to(marker_cluster)
            
            st_folium(m2, width=800, height=600)
            
            st.subheader("Raw Data")
            st.dataframe(species_df)
            
            # Download Button
            csv = species_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Species Data as CSV",
                data=csv,
                file_name=f"{selected_species.replace(' ', '_')}_sightings.csv",
                mime='text/csv',
            )
