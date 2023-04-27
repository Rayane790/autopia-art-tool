import streamlit as st
import pandas as pd  # pip install pandas openpyxl
import numpy as np
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import streamlit.components.v1 as com
import squarify


# Set the page configuration
st.set_page_config(
    page_title="ART-AutopiaWorld",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="auto"
)

def local_css(main):
    with open(main) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("/app/autopia-art-tool/StreamlitFinal/main.css")

# Add CSS styles to the page
hide_st_style = """
            <style>
            
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
# st.markdown(hide_st_style, unsafe_allow_html=True)

# Set dataframes
df_brand_model_year=pd.read_csv("/app/autopia-art-tool/StreamlitFinal/brand_model_year.csv",encoding='windows-1252')
df_brand_model=pd.read_csv("/app/autopia-art-tool/StreamlitFinal/brand_model.csv", encoding='windows-1252')
df_brand=pd.read_csv("/app/autopia-art-tool/StreamlitFinal/brand.csv", encoding='windows-1252')
df_brand_year=pd.read_csv("/app/autopia-art-tool/StreamlitFinal/brand_year.csv", encoding='windows-1252')    
df_year=pd.read_csv("/app/autopia-art-tool/StreamlitFinal/brand_year.csv", encoding='windows-1252')
# We will start filtering in the sheet brand_model_year
st.header("Here, you can see the distribution of a particular brand, based on its year of production, and  model. Choose a brand and get started!")

# Define the options for the brand selector
options = df_brand_model_year["brand"].unique()
# Create two columns for the selector and the slider
col1, col2 = st.columns(2)

    # Add the brand selector to the first column
brand_selected = col1.selectbox("**Select the brand:**", options=options)

filtered_df=df_brand_model_year.query("brand==@brand_selected")

    # Add the production year range slider to the second column
start_year, end_year = col2.slider('**Select a production year range**', min_value=int(filtered_df['productionyear'].min()), max_value=int(filtered_df['productionyear'].max()), value=(int(df_brand_model_year['productionyear'].min()), int(filtered_df['productionyear'].max())))
nb_brand=df_brand.query("brand==@brand_selected")

# Filter the DataFrame based on the selected production year range
filtered_df = filtered_df[(df_brand_model_year['productionyear'] >= start_year) & (df_brand_model_year['productionyear'] <= end_year)]


# Set respective values
index_value = nb_brand.loc[nb_brand['brand'] == brand_selected].index[0]
lower_bound = str(nb_brand.loc[index_value, 'L.B'])
upper_bound = str(nb_brand.loc[index_value, 'U.B'])
brand_name = str(nb_brand.loc[index_value, 'brand'])
status = str(nb_brand.loc[index_value, 'Status'])

# Write Count
st.write("According to our Smart Data, there are **between ", lower_bound, "to " , upper_bound, " ", brand_selected, " (", status, ")** Cars in Lebanon.")
if (len(filtered_df)<55):
    st.markdown("We can't display the graphs for your current selection. Either the range that you chose is too small, or this car is very rare.")
    st.write("Instead we will display a couple of charts about the top 5 most commom cars in Lebanon.")
    # Create the expanders for the charts
    expander1 = st.expander(("Distribution of car models by top 5 brands and models"), expanded=True)
    expander2 = st.expander("Distribution of top 5 years of production for top 5 brands", expanded=True)

    # Add the bar chart to the first expander
    with expander1:
        import plotly.express as px
        # Group the data by brand and cleaned_models
        df_grouped = df_brand_model_year.groupby(['brand', 'cleaned_model'], as_index=False)['rounded Count'].sum()

        # Get the top 5 brands based on the rounded count
        df_top5 = df_grouped.nlargest(5, 'rounded Count')

        df_grouped = df_brand_model_year.groupby(['brand', 'cleaned_model','productionyear'], as_index=False)['rounded Count'].sum()

        # Create a new DataFrame that contains the production data for the top 5 models of each of the top 5 brands
        df_filtered = pd.DataFrame()
        top_brands = df_grouped.groupby('brand', as_index=False)['rounded Count'].sum().nlargest(5, 'rounded Count')['brand'].tolist()
        for brand in top_brands:
            top_models = df_grouped[df_grouped['brand'] == brand].nlargest(5, 'rounded Count')['cleaned_model'].tolist()
            df_brand = df_grouped[(df_grouped['brand'] == brand) & (df_grouped['cleaned_model'].isin(top_models))]
            df_filtered = pd.concat([df_filtered, df_brand])

        # Define a list of custom colors for each brand
        brand_colors = ['#E01F24', '#fb6a4a ', '#fdd6c4', '#67000d', '#cb181d']

        # Create the TreeMap with custom brand colors
        fig = px.treemap(df_filtered, path=['brand', 'cleaned_model'], values='rounded Count', hover_data=['cleaned_model'],
                 color='brand', color_discrete_sequence=brand_colors)
        # Update the layout
        fig.update_layout(
        xaxis_title="Models",
        yaxis_title="Brand",
        font=dict(size=12)
        )
        fig.data[0].hovertemplate = '%{label}'
        # Display the TreeMap
        st.plotly_chart(fig, use_container_width=True)
        

    # Add the doughnut chart to the second expander
    with expander2:
        import pandas as pd
        import plotly.express as px

        # Filter the data to get the top 5 brands
        df_top_brands = df_filtered.groupby('brand', as_index=False)['rounded Count'].sum().nlargest(5, 'rounded Count')

        # Filter the data to get the top 5 years for each brand
        df_top_years = pd.DataFrame()
        for brand in df_top_brands['brand']:
            df_top_years = pd.concat([df_top_years, df_filtered[df_filtered['brand'] == brand].nlargest(5, 'rounded Count')])

        # Create the bar chart
        fig = px.bar(df_top_years, x='brand', y='rounded Count', color='productionyear', barmode='stack', text='productionyear', color_continuous_scale=px.colors.sequential.Reds, category_orders={'brand': df_top_years['rounded Count'].sort_values(ascending=False).index})

        # Update the layout
        fig.update_layout(
            xaxis_title="Brand",
            yaxis_title="",
            yaxis_tickformat=',.0f',
            yaxis_dtick=5000,
            height=500

        )


        # Display the stacked column chart
        st.plotly_chart(fig, use_container_width=True)
else:
    st.write("We will now portray the distribution of ", brand_selected, " cars based on **production year and model**. Feel free to adjust the year range and the models, and watch the magic happen")
    # Define the options for the brand selector
    options = df_brand_model_year["brand"].unique()

    # Create an expandable container for the charts

    options = df_brand_model_year["brand"].unique()

    # Create the expanders for the charts
    expander1 = st.expander(("Distribution of cars based on production year"), expanded=True)
    expander2 = st.expander("Distribution of cars based on model", expanded=True)

    # Add the bar chart to the first expander
    with expander1:
        fig1 = px.bar(filtered_df, x='productionyear', y='rounded Count' ,color='rounded Count')
        fig1.update_traces(marker_color='#E01F24')
        fig1.update_layout(
        xaxis_title="Brand",
        yaxis_title="Count"
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Add the doughnut chart to the second expander
    with expander2:
    # Calculate the percentage of each model
        filtered_df['Percent'] = 100 * filtered_df['rounded Count'] / filtered_df['rounded Count'].sum()

    # Filter out models with less than 5% of the total
        filtered_df = filtered_df[filtered_df['Percent'] >= 0.5]

    # Replace Null values with Unknown
        filtered_df["cleaned_model"].replace(np.nan, "Unknown", inplace=True)

        # Define the number of shades of red to use
        num_shades = len(filtered_df['cleaned_model'].unique())

        #   Define the base color as #E01F24
        base_color = '#E01F24'

        # Generate a color scale with shades of red
        import colorlover as cl
        colors = cl.scales['9']['seq']['Reds'][::-1][:num_shades-1]
        # Add the base color to the list of colors
        colors.append(base_color)

        # Create two columns for the multiselect and the bar chart
        col1, col2 = st.columns([1, 2])

        # Add the multiselect to the first column
        with col1:
            st.write("Please note that this list of models is by no means complete.")
            st.write("We chose to **not include models that constitute less than 5%** of the overall count of models to maintain the beauty of the chart")
            selected_models = st.multiselect("**Filter by selected models:**", filtered_df['cleaned_model'].unique(), default=filtered_df['cleaned_model'].unique(), key='multiselect2')

        # Check if at least one model is selected
            if not selected_models:
                st.warning("Please select at least one model.")
        # Check if there is only one model selected, and disable the option to unselect it
            elif len(selected_models) == 1:
                st.write(selected_models[0], "is the only remaining model.")
                st.write("Please select another model if you want to remove this one.")

        # Filter the data based on the selected models
        filtered_df3 = filtered_df[filtered_df['cleaned_model'].isin(selected_models)]

            # Add the bar chart to the second column
        with col2:
            # Create the doughnut chart
            fig2 = px.pie(filtered_df3, values='rounded Count', names='cleaned_model',
            color_discrete_sequence=colors, hole=0.5,
            template='plotly_dark', hover_name=['{:.2%}'.format(x) for x in filtered_df3['Percent']])
            fig2.update_traces(hovertemplate='%{label}: %{value}<br>Percentage:}')
            if(len(selected_models)>0):
                st.plotly_chart(fig2, use_container_width=True)
