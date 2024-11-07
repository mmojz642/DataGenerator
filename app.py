import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scipy.stats as stats

from custom_cdf import customCDF


def cdfHandler(dist, key:str=""):
    """Abstract handler of CDF distribution adjustment. Graph and sliders."""
    if not st.session_state.show_dist:
        return
    # Sliders in the left column
    with col1:
        st.write(f"Distribution {key}")
        sliders = [st.slider(f"Point {x}: {str(dist.x_points[x])}", 
                                min_value=0., 
                                max_value=1., 
                                value=x/dist.num_inputs, 
                                step=0.05,
                                key=key+str(x)) 
                        for x in range(0, len(dist))]
        
        # update the distribution
        for idx, yval in enumerate(sliders):
            dist.set_y_by_index(idx, yval)

    # Define x and y values based on slider inputs
    # Transpose graph given slider position
    x_values = dist.yvalues()
    y_values = dist.xvalues()
    
    # Create the line plot in the right column
    with col2:
        fig = go.Figure(
            data=go.Scatter(x=x_values, y=y_values, mode='lines+markers', marker=dict(size=10))
        )

        # Update layout for better appearance
        fig.update_layout(
            xaxis=dict(range=[0, 1], title="CDF"),
            yaxis=dict(range=[dist.max, dist.min], title="Range"),

        )
        # Display the plot
        st.plotly_chart(fig, key=key)

# Control seeing the distributions
if "show_dist" not in st.session_state:
    st.session_state.show_dist = False
def toggle_charts():
    st.session_state.show_dist = True

###
### Row 0: Enter distributions
###

# Create distributions
st.title("Generate distributions")

r0col1, r0col2 = st.columns(2)

with r0col1:
    # Input some categorical data
    items_text = st.text_area("List of Distributions", placeholder="Name, min, max\nCredit Score, 500, 850\nSalary, 0, 1000000")

with r0col2:
    # Allow display to get cleared
    r0col2_placeholder = st.empty()
    data = []

    # Parse the data into a dataframe
    for row in items_text.split("\n"):
        items = row.split(",")
        if len(items) == 3:
            items = [item.strip() for item in items]
            try:
                items[1] = float(items[1])
                items[2] = float(items[2])
                data.append(items)
                continue
            except:
                pass
        items = [items[0], 0, 1]
        data.append(items)
    data_df = pd.DataFrame(data, columns=["Name", "Min", "Max"])

    r0col2_placeholder.write("Current Distributions:")
    r0col2_placeholder.dataframe(data_df, width=200)

NUM_DISTS = data_df.shape[0]
if st.button('Update and Generate', on_click=toggle_charts):     
    # update column display
    with r0col2:
        r0col2_placeholder.write("Current Distributions:")
        r0col2_placeholder.write(data_df)

    st.title("Data Distribution Entry as CDFs")


# Display the CDFs and sliders for adjustment in two columns
col1, col2 = st.columns([1, 1])
if st.session_state.show_dist:
    distributions = {data_df["Name"].iloc[x]: customCDF(4, data_df["Min"].iloc[x], data_df["Max"].iloc[x]) for x in range(0, NUM_DISTS)}
    handlers = {x: cdfHandler(distributions[data_df["Name"].iloc[x]], key=data_df["Name"].iloc[x]) for x in range(0, NUM_DISTS)}


###
### Row 1: Enter correlations between distributions
###

# Display the correlation section
st.title("Correlation Data Entry")

r1col1, r1col2, r1col3 = st.columns(3)

with r1col1:
    num_variables = NUM_DISTS
    if st.session_state.show_dist:
        variable_names = [st.text_input(f"Variable", value=str(k)) for k in distributions.keys()]
    else:
        variable_names = [st.text_input(f"Variable {i+1} name", value=str(i+1)) for i in range(0, NUM_DISTS)]
# Create an empty DataFrame to store the correlations
correlation_matrix = pd.DataFrame([np.ones(NUM_DISTS)],index=variable_names, columns=variable_names)

with r1col2:
    # Enter correlation values
    st.write("Enter correlations (-1 to 1):")
    for i in range(num_variables):
        for j in range(i+1, num_variables):
            correlation_matrix.iloc[i, j] = st.number_input(f"Correlation btw. {variable_names[i]} and {variable_names[j]}", 
                                                            min_value=-1.0, 
                                                            max_value=1.0, 
                                                            step=0.01,
                                                            value=0.5)
            correlation_matrix.iloc[j, i] = correlation_matrix.iloc[i, j]  # Mirror the value

with r1col3:
    # Show the correlation matrix
    st.write("Correlation Matrix:")
    st.write(correlation_matrix)


###
### Row 3: Sample Data Generation
###

st.title("Generate Data")

rand = 0
transformed_data = []
uniform_data = []
if st.button('Click'): 
    # calculate means and std deviations
    means = [v.mean() for k, v in distributions.items()]
    stdevs = [v.stdev() for k, v in distributions.items()]

    # Make covariance matrix
    D = np.diag(stdevs)
    covariance_matrix = D @ correlation_matrix.to_numpy() @ D

    # Make sample data
    # Hardcoded as 10 for now

    sample_data = pd.DataFrame([], columns=[k for k in distributions.keys()])    
    for _ in range(0, 10):
        # Sample from a normal distribution to get the covariance right
        normal_data = np.random.multivariate_normal(means, covariance_matrix, size=1)
        
        # Get the standard normal data
        std_normal_data = normal_data.copy()
        for i, (_, dist) in enumerate(distributions.items()):
            std_normal_data[:, i] = (normal_data[:, i] - means[i]) / stdevs[i] 
        
        # Convert to uniform then back to custom distribution to get the range right
        uniform_data = stats.norm.cdf(std_normal_data)
        transformed_data = np.zeros_like(uniform_data)
        
        # Could make this vectorized: TODO: refactor
        for i, (_, dist) in enumerate(distributions.items()):
            transformed_data[:, i] = dist.inv_sample(uniform_data[:, i].item())

        sample_data = pd.concat([sample_data, pd.DataFrame(transformed_data, columns=[k for k in distributions.keys()])])    
    
    st.dataframe(sample_data, width=400)

    # Option to save or download the sample data
    st.download_button("Download sample data", sample_data.to_csv().encode('utf-8'), "sample_data.csv", "text/csv")
