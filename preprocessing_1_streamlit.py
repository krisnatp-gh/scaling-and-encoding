import streamlit as st
import pandas as pd
import numpy as np
from numerize.numerize import numerize # For formatting numbers in st.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import OneHotEncoder

# Page configuration
st.set_page_config(
                   page_title="Scaling and Encoding Class Demonstration",
                   layout="centered"
                   )

# Function to display binary conversion calculation
def calculate_binary_positive_integer(num):
    # Initiator
    list_latex_eqn = []
    dividend = num
    index = 0
    digits = ""

    while True:
        if dividend == 0:
            break
        
        quotient, remainder = divmod(dividend, 2)
        # Using boxed for highlighting the remainder
        latex_eqn = f"{dividend} \\div 2 &= {quotient} \\text{{ remainder }} \\boxed{{{remainder}}}"
        
        list_latex_eqn.append(latex_eqn)
        digits = str(remainder) + digits
        dividend = quotient
        index += 1

    # Clean LaTeX format
    formatted_latex_eqn = f"""
$$
\\begin{{aligned}}
{'\\\\'.join(list_latex_eqn)}
\\end{{aligned}}
\\quad\\quad
\\begin{{array}}{{c}}
\\Bigg\\uparrow \\\\
\\text{{{digits}}}
\\end{{array}}
$$
"""

    return formatted_latex_eqn
# Custom CSS for center alignment with left-aligned text
st.markdown("""
<style>
    .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

## <TO_DO_START>
# Load datasets
@st.cache_data
def load_datasets():
    disease_risk_df = pd.read_csv('Disease_Risk_No_Duplicates.csv')
    flight_df = pd.read_csv('flight_cancellation_with_agegroup.csv')
    car_insurance_df = pd.read_csv('Car_Insurance_Claims_dirty_V2.csv')
    return disease_risk_df, flight_df, car_insurance_df

disease_risk_df, flight_df, car_insurance_df = load_datasets()

# Section 1: Dataset and Column Selection
st.header("Dataset and Column Selection")

dataset_option = st.selectbox(
    "**Select Case**",
    ["Scaling", "Encoding"]
    # ["Scaling", "Encoding", "Imputing"]

)

# Determine which datasets are available for selection
if dataset_option == "Scaling":
    dataset_name = st.selectbox(
        "**Select Dataset**",
        ["Disease_Risk_No_Duplicates", "flight_cancellation_with_agegroup", "Car_Insurance_Claims_dirty_V2"]
    )
    
    if dataset_name == "Disease_Risk_No_Duplicates":
        current_df = disease_risk_df
    elif dataset_name == "flight_cancellation_with_agegroup":
        current_df = flight_df
    else:
        current_df = car_insurance_df
        current_df["ID"] = current_df["ID"].astype("object")
    
    numeric_cols = current_df.iloc[:,:-1].select_dtypes(include=['number']).columns.tolist()
    available_columns = numeric_cols

elif dataset_option == "Encoding":
    dataset_name = st.selectbox(
        "**Select Dataset**",
        ["flight_cancellation_with_agegroup", "Car_Insurance_Claims_dirty_V2"]
    )
    
    if dataset_name == "flight_cancellation_with_agegroup":
        current_df = flight_df
    else:
        current_df = car_insurance_df
    
    object_cols = current_df.iloc[:,:-1].select_dtypes(include=['object']).columns.tolist()
    available_columns = object_cols

else:  # Imputing
    current_df = car_insurance_df
    available_columns = current_df.iloc[:,:-1].loc[:,current_df.isna().any()].columns.tolist()



# Display first 5 rows of current dataset
st.data_editor(current_df.head(), num_rows="dynamic",  disabled=True)
column_selection = st.selectbox("**Select Feature**", available_columns)


X = current_df[[column_selection]]
y = current_df.iloc[:, -1] # target column
## Split data 
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=0,
                                                    stratify=y)


if dataset_option == "Encoding":
    # Display unique categories in selected column
    unique_categories = X_train[column_selection].dropna().unique()
    n_unique = len(unique_categories)
    
    if n_unique > 4:
        # Display first 2 categories and last one with ellipsis
        display_text = f"**{unique_categories[0]}**, **{unique_categories[1]}**, ..., **{unique_categories[-1]}**"
    else:
        # Display all categories
        display_text = ", ".join(f"**{str(cat)}**" for cat in unique_categories)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Unique Categories of `{column_selection}` (Train Data)")
        st.info(f"{display_text}")
    with col2:
        st.metric("Number of Unique Categories", n_unique)
    
elif dataset_option == "Imputing":
    # Display missing value information
    n_missing = current_df[column_selection].isna().sum()
    n_total = len(current_df)
    pct_missing = (n_missing / n_total) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Missing Values", f"{n_missing} ({pct_missing:.2f}%)")
    with col2:
        st.metric("Total Rows", n_total)


st.header("Method Selection")

with st.container():
    if dataset_option == "Scaling":
        scaler_method = st.selectbox(
            "Select Scaling Method",
            ["MinMaxScaler", "StandardScaler", "RobustScaler"]
        )

        # Display equation based on selected scaling method
        if scaler_method == "MinMaxScaler":
            st.latex(r"x^\text{new} = \frac{x-\text{min}(x_\text{train})}{\text{max}(x_\text{train}) - \text{min}(x_\text{train})}")
        elif scaler_method == "StandardScaler":
            st.latex(r"x^\text{new} = \frac{x-\text{mean}(x_\text{train})}{\text{std}(x_\text{train})}")
        else:  # RobustScaler
            st.latex(r"""\begin{aligned}
            x^\text{new} &= \frac{x-\text{Q}_1(x_\text{train})}{\text{Q}_3(x_\text{train}) - \text{Q}_1(x_\text{train})} \qquad \begin{array}{l}
            \text{Q}_1: \text{first quartile} \\
            \text{Q}_3: \text{third quartile}
            \end{array} \\
            \\
            &= \frac{x-\text{Q}_1(x_\text{train})}{\text{IQR}(x_\text{train})}
            \end{aligned}""")

        # Apply scaling
        if scaler_method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaler_method == "StandardScaler":
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()
        
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Data")
            
            # Display head of X_train and X_test side-by-side
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.write(f"**Train (first 3 rows)**")
                st.data_editor(X_train.head(3),
                               use_container_width=True,
                               hide_index=True,
                               num_rows="dynamic",  # Disable column editing by allowing adding new row
                               disabled=True)       # Disable editing the data

            with subcol2:
                st.write("**Test (first 3 rows)**")
                st.data_editor(X_test.head(3),
                               use_container_width=True,
                               hide_index=True,
                               num_rows="dynamic",  
                               disabled=True)
            
            # Display metrics based on scaling method
            if scaler_method == "MinMaxScaler":
                train_min = X_train[column_selection].min()
                train_max = X_train[column_selection].max()
                test_min = X_test[column_selection].min()
                test_max = X_test[column_selection].max()
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Train Min", numerize(float(train_min), 2))
                    st.metric("Train Max", numerize(float(train_max), 2))
                with metric_col2:
                    st.metric("Test Min", numerize(float(test_min), 2))
                    st.metric("Test Max", numerize(float(test_max), 2))
                    
            elif scaler_method == "StandardScaler":
                train_mean = X_train[column_selection].mean()
                train_std = X_train[column_selection].std()
                test_mean = X_test[column_selection].mean()
                test_std = X_test[column_selection].std()
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Train Mean", numerize(float(train_mean), 2))
                    st.metric("Train Std", numerize(float(train_std), 2))
                with metric_col2:
                    st.metric("Test Mean", numerize(float(test_mean), 2))
                    st.metric("Test Std", numerize(float(test_std), 2))
                    
            else:  # RobustScaler
                train_median = X_train[column_selection].median()
                train_q1 = X_train[column_selection].quantile(0.25)
                train_q3 = X_train[column_selection].quantile(0.75)
                train_iqr = train_q3 - train_q1
                
                test_median = X_test[column_selection].median()
                test_q1 = X_test[column_selection].quantile(0.25)
                test_q3 = X_test[column_selection].quantile(0.75)
                test_iqr = test_q3 - test_q1
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Train Median", numerize(float(train_median), 2))
                    st.metric("Train IQR", numerize(float(train_iqr), 2))
                with metric_col2:
                    st.metric("Test Median", numerize(float(test_median), 2))
                    st.metric("Test IQR", numerize(float(test_iqr), 2))
            
            # Create histogram with box plot
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.2, 0.8],
                vertical_spacing=0.05,
                specs=[[{"type": "box"}], [{"type": "histogram"}]]
            )
            
            fig.add_trace(
                go.Box(x=X_train[column_selection], name="", marker_color="steelblue"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(x=X_train[column_selection], name="Train", marker_color="steelblue", opacity=0.7),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text=column_selection, row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            fig.update_layout(height=400, showlegend=False, title_text="Unscaled Train Distribution")
            
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Scaled Data")
            
            # Display head of scaled data
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.write("**Train (first 3 rows)**")
                st.data_editor(X_train_scaled.head(3),
                               use_container_width=True,
                               hide_index=True,
                               num_rows="dynamic",  
                               disabled=True)
            with subcol2:
                st.write("**Test (first 3 rows)**")
                st.data_editor(X_test_scaled.head(3),
                               use_container_width=True,
                               hide_index=True,
                               num_rows="dynamic",  
                               disabled=True)            
            # Display metrics based on scaling method
            if scaler_method == "MinMaxScaler":
                train_min_scaled = X_train_scaled[column_selection].min()
                train_max_scaled = X_train_scaled[column_selection].max()
                test_min_scaled = X_test_scaled[column_selection].min()
                test_max_scaled = X_test_scaled[column_selection].max()
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Train Min", f"{train_min_scaled:.2f}")
                    st.metric("Train Max", f"{train_max_scaled:.2f}")
                with metric_col2:
                    st.metric("Test Min", f"{test_min_scaled:.2f}")
                    st.metric("Test Max", f"{test_max_scaled:.2f}")
                    
            elif scaler_method == "StandardScaler":
                train_mean_scaled = X_train_scaled[column_selection].mean()
                train_std_scaled = X_train_scaled[column_selection].std()
                test_mean_scaled = X_test_scaled[column_selection].mean()
                test_std_scaled = X_test_scaled[column_selection].std()
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Train Mean", f"{train_mean_scaled:.2f}")
                    st.metric("Train Std", f"{train_std_scaled:.2f}")
                with metric_col2:
                    st.metric("Test Mean", f"{test_mean_scaled:.2f}")
                    st.metric("Test Std", f"{test_std_scaled:.2f}")
                    
            else:  # RobustScaler
                train_median_scaled = X_train_scaled[column_selection].median()
                train_q1_scaled = X_train_scaled[column_selection].quantile(0.25)
                train_q3_scaled = X_train_scaled[column_selection].quantile(0.75)
                train_iqr_scaled = train_q3_scaled - train_q1_scaled
                
                test_median_scaled = X_test_scaled[column_selection].median()
                test_q1_scaled = X_test_scaled[column_selection].quantile(0.25)
                test_q3_scaled = X_test_scaled[column_selection].quantile(0.75)
                test_iqr_scaled = test_q3_scaled - test_q1_scaled
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Train Median", f"{train_median_scaled:.2f}")
                    st.metric("Train IQR", f"{train_iqr_scaled:.2f}")
                with metric_col2:
                    st.metric("Test Median", f"{test_median_scaled:.2f}")
                    st.metric("Test IQR", f"{test_iqr_scaled:.2f}")
            
            # Create histogram with box plot for scaled data
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.2, 0.8],
                vertical_spacing=0.05,
                specs=[[{"type": "box"}], [{"type": "histogram"}]]
            )
            
            fig.add_trace(
                go.Box(x=X_train_scaled[column_selection], name="", marker_color="coral"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(x=X_train_scaled[column_selection], name="Train Scaled", marker_color="coral", opacity=0.7),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text=column_selection, row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            fig.update_layout(height=400, showlegend=False, title_text="Scaled Train Distribution")
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif dataset_option == "Encoding":
        encoder_method = st.selectbox(
            "Select Encoding Method",
            ["OneHotEncoder", "BinaryEncoder", "OrdinalEncoder"]
        )
        if encoder_method == "OneHotEncoder":
            # Get unique categories from training data
            unique_categories = X_train[column_selection].dropna().unique()
            n_unique = len(unique_categories)
            
            # User selection for drop_first parameter
            st.subheader("OneHotEncoder Configuration")
            drop_first = st.checkbox(
                "Drop first category (to avoid multicollinearity)",
                value=False,
                help="When checked, drops the first category to avoid the dummy variable trap"
            )
            
            # Show encoding explanation
            st.subheader("OneHot Encoding Explanation")
            if drop_first:
                st.write(f"**Categories to encode:** {n_unique - 1} (dropping first category: `{unique_categories[0]}`)")
                st.info("Each remaining category gets its own binary column. The dropped category is represented by all zeros.")
            else:
                st.write(f"**Categories to encode:** {n_unique}")
                st.info("Each category gets its own binary column with value 1 when present, 0 otherwise.")
            
            # Show category mapping
            st.subheader("Category Mapping")
            if drop_first:
                categories_to_show = unique_categories[1:]
                mapping_text = f"**Dropped:** `{unique_categories[0]}` → represented by all zeros\n\n**Encoded categories:**"
            else:
                categories_to_show = unique_categories
                mapping_text = "**Encoded categories:**"
            
            st.markdown(mapping_text)
            
            # Display mapping in a clean format
            if len(categories_to_show) <= 5:
                for cat in categories_to_show:
                    st.markdown(f"- `{cat}` → gets column `{column_selection}_{cat}`")
            else:
                # Show first 2 and last 1
                for cat in categories_to_show[:2]:
                    st.markdown(f"- `{cat}` → gets column `{column_selection}_{cat}`")
                st.markdown("- ...")
                st.markdown(f"- `{categories_to_show[-1]}` → gets column `{column_selection}_{categories_to_show[-1]}`")
            
            # Perform OneHot encoding
            encoder = OneHotEncoder(drop='first' if drop_first else None, sparse_output=False, handle_unknown='ignore')
            
            X_train_encoded = encoder.fit_transform(X_train[[column_selection]])
            X_test_encoded = encoder.transform(X_test[[column_selection]])
            
            # Create column names
            if drop_first:
                encoded_columns = [f"{column_selection}_{cat}" for cat in unique_categories[1:]]
            else:
                encoded_columns = [f"{column_selection}_{cat}" for cat in unique_categories]
            
            X_train_encoded = pd.DataFrame(
                X_train_encoded,
                columns=encoded_columns,
                index=X_train.index
            )
            
            X_test_encoded = pd.DataFrame(
                X_test_encoded,
                columns=encoded_columns,
                index=X_test.index
            )
            
            # Display original and encoded data side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Data")
                st.write("**Train (first 5 rows)**")
                st.data_editor(X_train.head(5),
                                use_container_width=True,
                                hide_index=True,
                                num_rows="dynamic",
                                disabled=True)
            
            with col2:
                st.subheader("OneHot Encoded Data")
                
                n_encoded_cols = len(encoded_columns)
                
                # Determine which columns to display
                if n_encoded_cols <= 8:
                    display_train = X_train_encoded.head(5)
                else:
                    # Show first 2 columns, then ..., then last column
                    cols_to_show = [encoded_columns[0], encoded_columns[1], encoded_columns[-1]]
                    display_train = X_train_encoded[cols_to_show].head(5)
                    # Add ellipsis indicator in column name
                    display_train.columns = [display_train.columns[0], display_train.columns[1], f"... {display_train.columns[2]}"]
                
                st.write("**Train (first 5 rows)**")
                st.data_editor(display_train,
                            use_container_width=True,
                                hide_index=True,
                                num_rows="dynamic",
                                disabled=True)
                if n_encoded_cols > 8:
                    st.info(f"Showing first 2 and last of {n_encoded_cols} encoded columns")

        elif encoder_method == "BinaryEncoder":
            # Get unique categories from training data in order of appearance
            unique_categories = X_train[column_selection].dropna().unique()
            n_unique = len(unique_categories)
            
            # Calculate number of binary digits needed
            n_bits = int(np.ceil(np.log2(n_unique))) if n_unique > 1 else 1
            
            # Show the calculation for total number of unique categories
            st.subheader("Binary Encoding Calculation")
            st.write(f"**Number of unique categories:** {n_unique}")
            st.write(f"**Binary digits required:** {n_bits} (since $2^{{{n_bits}}} = {2**n_bits} \\geq {n_unique}$)")
            
            # Show conversion calculation for n_unique
            if n_unique > 1:
                st.write("**Converting number of categories to binary:**")
                st.markdown(calculate_binary_positive_integer(n_unique), unsafe_allow_html=True)
            
            # Create mapping dictionary and display mappings
            st.subheader("Category to Binary Mapping")
            category_to_int = {cat: idx for idx, cat in enumerate(unique_categories)}
            
            # Build LaTeX string for mappings
            latex_mappings = []
            for category, int_val in category_to_int.items():
                binary_val = format(int_val, f'0{n_bits}b')
                latex_mappings.append(
                    f"\\text{{{category}}} \\xrightarrow{{\\text{{category order}}}} {int_val} "
                    f"\\xrightarrow{{\\text{{binary}}}} {binary_val}"
                )
            
            # Display mappings
            if len(latex_mappings) <= 5:
                mapping_display = " \\\\\n".join(latex_mappings)
            else:
                # Show first 3, ellipsis, and last one
                mapping_display = " \\\\\n".join(latex_mappings[:3]) + \
                                " \\\\\n\\vdots \\\\\n" + latex_mappings[-1]
            
            st.latex(f"\\begin{{aligned}}\n{mapping_display}\n\\end{{aligned}}")
            
            # Perform binary encoding
            def binary_encode_column(df, column, category_mapping, n_bits):
                """Encode a categorical column to binary representation"""
                encoded_df = pd.DataFrame(index=df.index)
                
                for i in range(n_bits):
                    col_name = f"{column}_{i}"
                    encoded_df[col_name] = df[column].map(
                        lambda x: int(format(category_mapping.get(x, 0), f'0{n_bits}b')[i]) 
                        if pd.notna(x) else np.nan
                    )
                
                return encoded_df
            
            X_train_encoded = binary_encode_column(X_train, column_selection, category_to_int, n_bits)
            X_test_encoded = binary_encode_column(X_test, column_selection, category_to_int, n_bits)
            
            # Display original and encoded data side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Data")
                st.write("**Train (first 3 rows)**")
                st.data_editor(X_train.head(3),
                                use_container_width=True,
                                hide_index=True,
                                num_rows="dynamic",
                                disabled=True)
            
            with col2:
                st.subheader("Binary Encoded Data")
                
                # Determine which columns to display
                if n_bits <= 8:
                    display_train = X_train_encoded.head(5)
                else:
                    # Show first 2 columns, then ..., then last column
                    cols_to_show = list(X_train_encoded.columns[:2]) + [X_train_encoded.columns[-1]]
                    display_train = X_train_encoded[cols_to_show].head(5)
                    # Rename columns to indicate there are more
                    display_train.columns = [display_train.columns[0], display_train.columns[1], f"... {display_train.columns[2]}"]
                
                st.write("**Train (first 3 rows)**")
                st.data_editor(display_train.head(3),
                               use_container_width=True,
                                hide_index=True,
                                num_rows="dynamic",
                                disabled=True)
                if n_bits > 8:
                    st.info(f"Showing first 2 and last of {n_bits} binary columns")

        else:
            st.subheader("Ordinal Mapping")
            unique_categories = current_df[column_selection].dropna().unique()
            
            # Create ordinal mapping inputs
            # Create a mapping dictionary from user inputs
            ordinal_mapping = {}
            for category in unique_categories:
                ordinal_mapping[category] = st.number_input(
                    f"Order for '{category}'",
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"ordinal_{category}"
                )
            
            # Apply ordinal encoding
            X_train_encoded = X_train.copy()
            X_test_encoded = X_test.copy()
            
            X_train_encoded[column_selection] = X_train[column_selection].map(ordinal_mapping)
            X_test_encoded[column_selection] = X_test[column_selection].map(ordinal_mapping)
            
            # Display the mapping table
            st.subheader("Ordinal Mapping Table")
            mapping_df = pd.DataFrame({
                'Category': list(ordinal_mapping.keys()),
                'Ordinal Value': list(ordinal_mapping.values())
            }).sort_values('Ordinal Value')
            
            st.dataframe(mapping_df, use_container_width=True, hide_index=True)
            
            # Display original and encoded data side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Data")
                st.write("**Train (first 5 rows)**")
                st.data_editor(X_train.head(5),
                                use_container_width=True,
                                hide_index=True,
                                num_rows="dynamic",
                                disabled=True)
                
            with col2:
                st.subheader("Ordinal Encoded Data")
                st.write("**Train (first 5 rows)**")
                st.data_editor(X_train_encoded.head(5),
                            use_container_width=True,
                                hide_index=True,
                                num_rows="dynamic",
                                disabled=True)
                

    else:  # Imputing
        column_dtype = current_df[column_selection].dtype
        
        if pd.api.types.is_numeric_dtype(column_dtype):
            imputer_method = st.selectbox(
                "Select Imputing Method",
                ["SimpleImputer", "IterativeImputer", "KNNImputer"]
            )
            
            if imputer_method == "SimpleImputer":
                strategy = st.selectbox(
                    "Select Strategy",
                    ["mean", "median", "mode", "constant"]
                )
                
                if strategy == "constant":
                    constant_value = st.number_input(
                        "Enter constant value",
                        value=0.0,
                        step=0.1
                    )
                    
                    if not isinstance(constant_value, (int, float)):
                        st.warning("Invalid input: Please enter a numeric value")
        
        else:  # Object column
            imputer_method = st.selectbox(
                "Select Imputing Method",
                ["SimpleImputer"]
            )
            
            strategy = st.selectbox(
                "Select Strategy",
                ["mode", "constant"]
            )
            
            if strategy == "constant":
                constant_value = st.text_input("Enter constant value")
                
                if constant_value == "":
                    st.warning("Invalid input: Please enter a non-empty string")