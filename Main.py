
from sklearn.metrics import mean_absolute_percentage_error
import plotly.express as px
from prophet import Prophet
from statsmodels.stats.proportion import proportions_ztest

from IPython.display import display, HTML, Markdown

from IPython.display import HTML
from datetime import timedelta
import seaborn as sns

import matplotlib.pyplot as plt
import altair as alt

from streamlit import components
import streamlit as st
import pandas as pd
import nump as np
import streamlit.components.v1 as comp





from html_code import render_dataframe_with_tooltips

def authenticate(username, password):

    return username == "Bizinsight" and password == "Bizinsight@2024"

# Define the login page content
def login_page():
    # Add CSS to set the background image
    page_bg_img = '''
        <style>
        [data-testid="stAppViewContainer"]
        {
        background-image: url("https://tse3.mm.bing.net/th/id/OIG1.mx9Iuw7cQ.l9_IOhuHDs?pid=ImgGn");
        background-size: cover;
        }
        </style>
        '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


    # Write the login form
    st.write("# Login")

    # Get username and password input from the user
    st.markdown(
        """
        <style>
        /* Style for text input boxes */
        input[type="text"],
        input[type="password"] {
            background-color: #D3D3D3; /* Set background color to red */
            color: #000000; /* Set text color to white */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Get username and password input from the user
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # If the user clicks the "Login" button
    if st.button("Login"):
        # Authenticate the user
        if authenticate(username, password):
            # Store the authentication state in session state
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password")

# Define the main content of the application
def main():
    # Display the login page if the user is not logged in
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login_page()
        return
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    global graphNumber, appendReal, PATTERN_CHECKS_RESULTS, container_Brand_volume, container_data_uniqueness , output_file

    st.set_page_config(layout="wide")

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F2F2F2; /* Blue color */
        }
        </style>
        """,
        unsafe_allow_html=True)

    # st.markdown("""<style>.stApp {padding: 0;margin: 0;}.stImage > div:first-child {padding: 0;}</style>""",unsafe_allow_html=True)

    st.image("logoo.png")

    # st.image(r"logo.jpg", width=200)

    config_df = pd.read_csv('ConfigAll_Biz.csv')

    # Select only the first six columns
    # config_df_first_six = config_df.iloc[:, :6]
    st.markdown(f"<div style='margin-bottom: -70px; font-weight:bold; color:black; text-align:left;'>DQ Monitoring Rules Configuration</div>",unsafe_allow_html=True)
    st.write(' ')
    # Display the DataFrame
    # with st.expander(""):
        
    #     show_full_table = st.checkbox("Show full table")


    #     if show_full_table:

    #         st.dataframe(config_df, hide_index=True)
    #     else:
    #         # Select only the first six columns
    #         config_df_first_six = config_df.iloc[:, :8]
    #         config_df_first_six = config_df_first_six.reset_index(drop=True)

    #         st.dataframe(config_df_first_six, hide_index=True)
    with st.expander(""):
        df_edited = st.data_editor(config_df, hide_index=True)
        df_edited.to_csv('ConfigAll_Biz.csv', index=False)




    st.markdown(f"<div style='margin-bottom: -70px; font-weight:bold; color:black; text-align:center;'>Data Quality Monitoring</div>",unsafe_allow_html=True)

    # Display the dropdown box
    selected_cell = st.selectbox("", config_df['Table Name'], format_func=lambda x: x,
                                 help="Select the table name", key="selectbox")

    # Apply CSS to add border
    st.markdown(
        f"""<style>
            div[data-baseweb="select"] {{
                border: 1.5px solid black ;
                border-radius: .25rem;
                
            }}
        </style>""",
        unsafe_allow_html=True
    )

    # Check if a table is selected
    if selected_cell is not None:
    
        
        # Get the index of the selected row
        selected_row_index = config_df[config_df['Table Name'] == selected_cell].index[0]

        # Get the ID number corresponding to the selected row
        selected_id = config_df.iloc[selected_row_index]['ID']
        selected_File_Name = config_df.iloc[selected_row_index]['File_Name']
        selected_brand_Name = config_df.iloc[selected_row_index]['brands_list']
        selected_rca_Name = config_df.iloc[selected_row_index]['RCA Analysis Columns']
    
        selected_brand_Name = str(selected_brand_Name)  # Convert to string if it's not already
        brands_list = selected_brand_Name.split(";")
        selected_rca_Name = str(selected_rca_Name)  # Convert to string if it's not already
        rca_list = selected_rca_Name.split(";")




        # Print the selected ID number
        number = selected_id
        #(config_df['ID'] == number).any()

        if (config_df['ID'] == number).any():
       
            def prep_and_split_data(df, threshold_date):
                df['is_weekend'] = df.ds.dt.day_name().isin(['Saturday', 'Sunday'])
                threshold_date = pd.to_datetime(threshold_date)
                mask = df['ds'] < threshold_date
                # Split the data and select `ds` and `y` columns.
                df_train = df[mask]
                df_test = df[~ mask]
                return df_train, df_test

            def check_freshness(df, date, columns_for_freshness):
                global freshnessColumns, freshnessDf
                #st.write(f"Running Data Freshness check for {date}")
                for column in columns_for_freshness:
                    if df.loc[df['ds'] == date, column].shape[0] == 0:
                        # st.write(f"No data for {date} in column {column}")
                        new_row = [column, '', 'Fail', 'No Value in column is having Latest Data']
                        new_row_df = pd.DataFrame([new_row], columns=freshnessDf.columns)
                        freshnessDf = pd.concat([freshnessDf, new_row_df])
                        pass

                    else:
                        # st.write("------------------------------------------------------------------------------")
                        # bold_text = f"**Data is fresh. {column} column is having the latest data for {date}**"
                        # pass
                        # df = df.append(new_row_dict, ignore_index=True)

                        # st.write(bold_text)
                        uniqueValues = df[column].unique()
                        for val in uniqueValues:
                            if df.loc[(df['ds'] == date) & (df[column] == val)].shape[0] == 0:
                                # st.write(f"No data for {date} for {val} in column {column}")
                                new_row = [column, val, 'Fail', f"No data for {date} for {val} in column {column}"]
                                new_row_df = pd.DataFrame([new_row], columns=freshnessDf.columns)
                                # freshnessDf = freshnessDf.concat([freshnessDf, new_row_df], ignore_index=True)
                                # freshnessDf = freshnessDf + new_row_df
                                freshnessDf = pd.concat([freshnessDf, new_row_df])
                            else:
                                # st.write(f"{val}: Data freshness timeline does not see any issues and having latest data for {date}")
                                new_row = [column, val, 'Pass', '']
                                new_row_df = pd.DataFrame([new_row], columns=freshnessDf.columns)
                                freshnessDf = pd.concat([freshnessDf, new_row_df])

            def check_zeros(df, output_file):
                results = []
                for col in df.columns:
                    if (df[col] == 0).any():
                        results.append((col, "Zero", "Column contains Zeros"))
                    else:
                        results.append((col, "Not Zero", ""))
                result_df = pd.DataFrame(results, columns=["Dimension", "Check", "Comments"])
                result_df['Status'] = result_df['Check'].apply(lambda x: 'Fail' if x == 'Zero' else 'Pass')
                result_df = result_df[["Dimension", "Status", "Comments"]]
                result_df.to_csv(output_file, index=False)


            def check_data_volume_new(df, date, columns_for_volume):
                bold_text = f"**Running Data Volume check for {date}**"
                st.write(bold_text)
                df = df
                df_historical_dates = df.copy()
                for column_to_check in columns_for_volume:
                    count_historical = df_historical_dates.groupby([column_to_check, 'ds']).size().reset_index(
                        name='Count')
                    forecast, mape = get_table_volume_forecast(count_historical, date_col='ds', prediction_col='Count')
                    forecast = forecast[forecast.ds == date]
                    yhat_lower = forecast.yhat_lower.values[0]
                    yhat_upper = forecast.yhat_upper.values[0]
                    data_volume = df.loc[df.ds == date].shape[0]
                    # assert yhat_lower <= data_volume <= yhat_upper, f"Data volume is {data_volume}, and is outside the expected range of:\n{yhat_lower} to {yhat_upper}"
                    st.write(
                        f"For {column_to_check} Data volume is {data_volume}. It is within the expected range of {yhat_lower} to {yhat_upper}")

            def get_table_volume_forecast(df, date_col, prediction_col):
                df = df.groupby(date_col).sum(prediction_col).reset_index()
                df['y'] = df[prediction_col]
                df['ds'] = pd.to_datetime(df[date_col])
                daily_train, daily_test = prep_and_split_data(df, THRESHOLD_DATE)
                model_baseline = Prophet()
                model_baseline.fit(daily_train)
                forecast = model_baseline.predict(daily_test)
                performance_baseline = pd.merge(daily_test, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                                                on='ds')
                performance_baseline = performance_baseline[~performance_baseline.is_weekend]
                performance_baseline_MAPE = mean_absolute_percentage_error(performance_baseline['y'],
                                                                           performance_baseline['yhat'])
                return forecast, performance_baseline_MAPE

            def check_nulls_new(df, date, column, columns_for_not_nulls, baseline, baseline_obs):
                fig = check_missing_values_new(df.loc[df.ds == date], column, columns_for_not_nulls, baseline,
                                               baseline_obs)
                # st.pyplot(fig)

            def check_null_changes_new(df, baseline, baseline_obs, columns_for_not_nulls):
                global notnullDf
                null_count_df = df.isnull().sum().reset_index()
                null_count_df.columns = ['Column', 'Null Count']
                df_obs = df.shape[0]
                col_dict = {}
                # for col in baseline['Column'].values.tolist():

                bold_text = f"**Initiating NUll checks for columns. {columns_for_not_nulls}**"
                # st.write(bold_text)
                for col in columns_for_not_nulls:
                    if baseline[baseline['Column'] == col]['Null Count'].values[0] == 0:
                        if null_count_df[null_count_df['Column'] == col]['Null Count'].values[0] != 0:
                            # st.write(f"{col} column has nulls where it didn't before.")
                            col_dict[col] = True
                            new_row = [col, 'Fail', 'New NULL Values detected']
                            new_row_df = pd.DataFrame([new_row], columns=notnullDf.columns)
                            notnullDf = pd.concat([notnullDf, new_row_df])
                        else:
                            # st.write(f"{col} column doesn't have nulls")
                            col_dict[col] = False
                            new_row = [col, 'Pass', '']
                            new_row_df = pd.DataFrame([new_row], columns=notnullDf.columns)
                            notnullDf = pd.concat([notnullDf, new_row_df])
                    else:
                        null_count = null_count_df[null_count_df['Column'] == col]['Null Count'].values[0]
                        baseline_count = baseline[baseline['Column'] == col]['Null Count'].values[0]
                        count = [baseline_count, null_count]
                        nobs = [baseline_obs, df_obs]
                        pval = proportions_ztest(count, nobs)[1]
                        if pval <= .05:
                            # st.write(f"{col} column has nulls present, and they are significantly different than historical null proportion.")
                            col_dict[col] = True
                            new_row = [col, 'Fail', 'Significantly different from historical trend']
                            new_row_df = pd.DataFrame([new_row], columns=notnullDf.columns)
                            notnullDf = pd.concat([notnullDf, new_row_df])
                        else:
                            # st.write(f"{col} column has nulls present, but they are not significantly different than historical null proportion.")
                            new_row = [col, 'Pass.', 'Not significantly different from historical trend']
                            new_row_df = pd.DataFrame([new_row], columns=notnullDf.columns)
                            notnullDf = pd.concat([notnullDf, new_row_df])
                            col_dict[col] = False
                return null_count_df, col_dict

            def check_uniqueness_new(df, date, column, columns_to_check_uniqueness):
                global uniqueDf

                bold_text = f"**Checking Uniqueness for columns {columns_to_check_uniqueness}**"
                # st.write(bold_text)

                baseline = df
                date_df = df.loc[df.ds == date]
                # columns_to_check = set(date_df.columns).intersection(baseline.columns)
                for col in columns_to_check_uniqueness:
                    baseline_is_unique = baseline[col].is_unique
                    date_df_is_unique = date_df[col].is_unique
                    if baseline_is_unique == date_df_is_unique == True:
                        st.write(f'{col} column is still unique')
                        new_row = [col, 'Pass', '']
                        new_row_df = pd.DataFrame([new_row], columns=uniqueDf.columns)
                        uniqueDf = pd.concat([uniqueDf, new_row_df])


                    elif baseline_is_unique != date_df_is_unique:
                        st.write(f"{col}'s unique value was {baseline_is_unique}, but is now {date_df_is_unique}")
                        new_row = [col, 'Fail', 'Not Proportionate with historical trend']
                        new_row_df = pd.DataFrame([new_row], columns=uniqueDf.columns)
                        uniqueDf = pd.concat([uniqueDf, new_row_df])

                    else:
                        # st.write(f"{col}'s values are not unique and it is in proportinate with Historical not unique trend")
                        new_row = [col, 'Pass.', 'Proportionate with historical Unique trend']
                        new_row_df = pd.DataFrame([new_row], columns=uniqueDf.columns)
                        uniqueDf = pd.concat([uniqueDf, new_row_df])

                        # print(f"{col} is not unique")

            def check_missing_values_new(df, column, columns_for_not_nulls, baseline, baseline_obs):
                null_count_df, col_dict = check_null_changes_new(df, baseline, baseline_obs, columns_for_not_nulls)
                color_df = pd.DataFrame(col_dict.items(), columns=['Column', 'isSignifcantNullChange'])
                df_to_plot = pd.merge(null_count_df, color_df, on='Column')
                fig = px.bar(data_frame=df_to_plot,
                             x='Null Count',
                             y='Column',
                             text='Null Count',
                             title='Null Values per Column',
                             color='isSignifcantNullChange')
                return fig

            def check_columns_new(df, baseline, baseline_obs, date, columns_for_not_nulls, columns_to_check_uniqueness):
                column = None
                check_nulls_new(df, date, column, columns_for_not_nulls, baseline, baseline_obs)
                check_uniqueness_new(df, date, column, columns_to_check_uniqueness)

            def get_dimension_changes(filtered_data_8_weeks, metric_column_name, columnName, metric_rca_val,failureType):
                global RCAdf
                columns_columnDimension = [columnName, 'ds', metric_column_name]
                filtered_data_columnDimension = filtered_data_8_weeks[columns_columnDimension]

                filtered_data_columnDimension[metric_column_name] = pd.to_numeric(
                    filtered_data_columnDimension[metric_column_name], errors='coerce')

                filtered_data_columnDimension = \
                    filtered_data_columnDimension.groupby(['ds', columnName], as_index=False)[metric_column_name].sum()

                grouped_data_columnDimension = filtered_data_columnDimension.groupby('ds')[
                    metric_column_name].sum().reset_index()

                filtered_data_columnDimension = pd.merge(filtered_data_columnDimension, grouped_data_columnDimension,
                                                         on='ds', suffixes=('', '_total'))

                metricTotal = metric_column_name + '_total'

                filtered_data_columnDimension[metricTotal] = pd.to_numeric(filtered_data_columnDimension[metricTotal],
                                                                           errors='coerce')
                filtered_data_columnDimension['pct'] = filtered_data_columnDimension[metric_column_name] / \
                                                       filtered_data_columnDimension[metricTotal] * 100

                filtered_data_columnDimension['ds'] = pd.to_datetime(filtered_data_columnDimension['ds'])

                most_recent_week_start = filtered_data_columnDimension['ds'].max() - pd.DateOffset(
                    days=filtered_data_columnDimension['ds'].max().dayofweek)

                filtered_data_recent = filtered_data_columnDimension[
                    filtered_data_columnDimension['ds'] >= most_recent_week_start]
                filtered_data_history = filtered_data_columnDimension[
                    filtered_data_columnDimension['ds'] < most_recent_week_start]

                filtered_data_history = filtered_data_history.groupby([columnName], as_index=False)['pct'].mean()

                filtered_data_recent = filtered_data_recent[[columnName, 'pct']]

                filtered_data_final = pd.merge(filtered_data_recent, filtered_data_history, on=columnName,suffixes=('_recent', '_history'))

                filtered_data_final['abs'] = filtered_data_final['pct_recent'] - filtered_data_final['pct_history']

                if failureType == 'Upper':

                    filtered_data_final = filtered_data_final.sort_values(by='abs', ascending=False)
                else:
                    filtered_data_final = filtered_data_final.sort_values(by='abs', ascending=True)


                #st.write(filtered_data_final)

                filtered_data_columnDimension = filtered_data_final.head(1)
                # filtered_data_columnDimension = filtered_data_final.copy()

                filtered_data_columnDimension = filtered_data_columnDimension.rename(columns={columnName: 'List_Value'})

                filtered_data_columnDimension['Column Name'] = columnName

                filtered_data_columnDimension['Metric Val'] = metric_rca_val

                filtered_data_columnDimension['change_text'] = filtered_data_columnDimension.apply(change_text, axis=1)


                if len(filtered_data_columnDimension) > 0:
                    for index, row in filtered_data_columnDimension.iterrows():
                        return row['change_text'], abs(row['abs'])
                else: 
                    return 'No Major changes detected', 0

                # failureText = ''
                # for text_val in filtered_data_columnDimension['change_text'].unique():
                #     if failureText:
                #         failureText = failureText + '\n' + text_val
                #     else:
                #         failureText = text_val
                # return failureText

                #return filtered_data_columnDimension['change_text'], row['abs']


            def perform_root_cause(latest_8_weeks_data, metric_column_name, columns_for_RCA_analysis, metric_rca_val,failureType):

                textOpFinal = ''
                val1 = 0
                for column in columns_for_RCA_analysis:
                    textOp, val = get_dimension_changes(latest_8_weeks_data, metric_column_name, column, metric_rca_val,failureType)
                    if val > val1:
                        textOpFinal = textOp
                        val1 = val
                    

                    


                    # if textOpFinal:
                    #     textOpFinal = textOpFinal + '\n' + '\n' + textOp
                    # else:
                    #     textOpFinal = textOp
                return textOpFinal

            def custom_rca_for_each_week(max_date, failed_records, METRIC_COLUMN, metric_rca_val,failureType):

                # st.write('*****************************************************************************')

                rcaDate = str(max_date).replace(' 00:00:00', '')
                bold_text = f"**RCA analysis for Anomaly {rcaDate}**"
                # st.write(bold_text)

                start_date_8_weeks_ago = max_date - timedelta(weeks=8)
                latest_8_weeks_data = failed_records[(failed_records['ds'] > start_date_8_weeks_ago) & (failed_records['ds'] < max_date)].copy()
                textDict = {'d1': 'dval'}
                RCAdf = pd.DataFrame()

                RCAdf = pd.DataFrame(columns=['List_Value', 'pct_recent', 'pct_history', 'abs', 'Column Name', 'Metric Val'])
                columns_for_RCA_analysis = rca_list

                RCADF = perform_root_cause(latest_8_weeks_data, METRIC_COLUMN, columns_for_RCA_analysis, metric_rca_val,failureType)

                return RCADF

            def FILTER_DATA_FOR_EACH_COMBINATION(CURRENT_COMBINATION, LIST_OF_AGGREGATE_COLUMNS, EXTRACTED_DATA_FROM_DB,
                                                 METRIC_COLUMN, rca_text, metric_val):

                FILTERED_DATA_DF = EXTRACTED_DATA_FROM_DB.copy()

                FILTER_CONDITION = None
                for col, val in zip(LIST_OF_AGGREGATE_COLUMNS, CURRENT_COMBINATION):
                    CONDITION = f"(FILTERED_DATA_DF['{col}'] == '{val}')"
                    if FILTER_CONDITION is None:
                        FILTER_CONDITION = CONDITION
                    else:
                        FILTER_CONDITION = FILTER_CONDITION + ' & ' + CONDITION

                FILTERED_DATA_FOR_MODEL = FILTERED_DATA_DF[eval(FILTER_CONDITION)].copy()

                RUN_TIME_SERIES_ANALYSIS(FILTERED_DATA_FOR_MODEL, CURRENT_COMBINATION, LIST_OF_AGGREGATE_COLUMNS,
                                         METRIC_COLUMN, rca_text, metric_val)

            def PERFORM_DATA_PATTERN_CHECKS(metric, aggreg, DATE_COLUMN_NAME, METRIC_COLUMN_NAME, VALUE_COLUMN_NAME,
                                            FILTERED_DATA_FOR_CURRENT_METRIC, PIVOT_FLAG):
               

                # st.write(aggreg)
                global PATTERN_CHECKS_RESULTS

                aggregateDimension = aggreg
                metricColumns = list(set([METRIC_COLUMN_NAME.strip(), VALUE_COLUMN_NAME.strip()]))

                # Selecting the data for required columns, if there is a sublist of columns if enters the if clause

                if isinstance(aggreg, list):
                    columnName = '_'.join(aggreg)
                    filtered_data = FILTERED_DATA_FOR_CURRENT_METRIC.copy()
                    filtered_data[columnName] = filtered_data[aggreg[0]] + '_' + filtered_data[aggreg[1]]
                    AGGREGARTE_NEW_COLUMN = columnName
                    AGGREGARTE_COLUMNS_LIST = aggreg + [DATE_COLUMN_NAME] + [AGGREGARTE_NEW_COLUMN] + metricColumns
                    selected_data = filtered_data[AGGREGARTE_COLUMNS_LIST].copy()
                else:
                    filtered_data = FILTERED_DATA_FOR_CURRENT_METRIC.copy()
                    AGGREGARTE_NEW_COLUMN = aggreg
                    AGGREGARTE_COLUMNS_LIST = [aggreg] + [DATE_COLUMN_NAME] + metricColumns
                    selected_data = filtered_data[AGGREGARTE_COLUMNS_LIST].copy()

                # Renaming the Date column to DS and Metric column to Y
                selected_data.rename(columns={DATE_COLUMN_NAME: 'ds', VALUE_COLUMN_NAME: 'y'}, inplace=True)
                selected_data['y'] = selected_data['y'].fillna(0).astype(int)

                if PIVOT_FLAG != 'Yes':
                    aggregated_data = selected_data.groupby([METRIC_COLUMN_NAME, AGGREGARTE_NEW_COLUMN, 'ds']).agg(
                        {'y': 'sum'}).reset_index()
                    aggregated_data = aggregated_data[[METRIC_COLUMN_NAME, AGGREGARTE_NEW_COLUMN, 'y', 'ds']]
                else:
                    aggregated_data = selected_data.groupby([AGGREGARTE_NEW_COLUMN, 'ds']).agg(
                        {'y': 'sum'}).reset_index()
                    aggregated_data = aggregated_data[[AGGREGARTE_NEW_COLUMN, 'y', 'ds']]

                try:
                    aggregated_data['ds'] = aggregated_data['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))
                except Exception as e:
                    pass
                aggregated_data['ds'] = pd.to_datetime(aggregated_data['ds'], format='%Y-%m-%d', errors='coerce')

                aggregated_data = aggregated_data[aggregated_data['ds'] > '2023-01-01']

                model_data = aggregated_data.copy()
                model_data['ds'] = pd.to_datetime(model_data['ds'], format='%Y-%m-%d')
                model_data['y'] = model_data['y'].astype(int)
                total_values = model_data.groupby('ds')['y'].sum().reset_index()
                merged_data = pd.merge(model_data, total_values, on='ds', suffixes=('', '_total'))
                merged_data[['y', 'y_total']] = merged_data[['y', 'y_total']].astype(float)
                merged_data['Contribution'] = (merged_data['y'] / merged_data['y_total']) * 100

                if PIVOT_FLAG != 'Yes':
                    merged_data['Key'] = merged_data[METRIC_COLUMN_NAME] + '_' + merged_data[AGGREGARTE_NEW_COLUMN]
                else:
                    merged_data['Key'] = merged_data[AGGREGARTE_NEW_COLUMN]

                if metric == None:
                    metric = VALUE_COLUMN_NAME

                latest_5_weeks = merged_data['ds'].nlargest(5)
                latest_5_weeks_data = merged_data[merged_data['ds'].isin(latest_5_weeks)]

                # Group by 'Product' and filter for products where any contribution is less than 6%
                filtered_data = latest_5_weeks_data.groupby('Key').filter(
                    lambda x: (x['Contribution'] < 6).any()).copy()

                # get buckets less than 6%
                # latest_week = merged_data['ds'].max()
                # latest_week_data = merged_data[merged_data['ds'] == latest_week]
                # filtered_data = latest_week_data[latest_week_data['Contribution'] < 6].copy()
                product_names_list = filtered_data['Key'].tolist()

                merged_data.loc[merged_data['Key'].isin(product_names_list), 'Key'] = 'Others'

                # merged_data.to_csv('merged_data2.csv')

                merged_data[['Contribution']] = merged_data[['Contribution']].astype(float)
                if PIVOT_FLAG != 'Yes':
                    merged_data = merged_data.groupby([METRIC_COLUMN_NAME, AGGREGARTE_NEW_COLUMN, 'ds', 'Key']).agg(
                        {'Contribution': 'sum'}).reset_index()
                    # merged_data = merged_data.groupby(['METRIC', 'PRODUCT', 'ds','Key']).agg({'Contribution': 'sum'}).reset_index()


                else:
                    merged_data = merged_data.groupby([AGGREGARTE_NEW_COLUMN, 'ds', 'Key']).agg(
                        {'Contribution': 'sum'}).reset_index()

                unique_keys = list(merged_data['Key'].unique())

                # Data filtering and manipulation
                key_filter_data = merged_data[merged_data['Key'].isin(unique_keys)]
                key_filter_data = key_filter_data[['Key', 'ds', 'Contribution']]
                key_filter_data.rename(columns={'Contribution': 'y'}, inplace=True)
                key_filter_data['y'] = pd.to_numeric(key_filter_data['y'], errors='coerce') / 100

                key_filter_data['ds'] = pd.to_datetime(key_filter_data['ds'], errors='coerce', dayfirst=True,
                                                       format='%Y-%m-%d')

                # Get the latest 52 weeks of data

                max_date = key_filter_data['ds'].max()

                start_date_52_weeks_ago = max_date - timedelta(weeks=108)

                # key_filter_data = key_filter_data[key_filter_data['ds'] > '2023-09-25']

                if len(key_filter_data) < 1:
                    # print('No data available')
                    return 200

                key_filter_data.to_csv('key_filter_data.csv')
                for i in unique_keys:

                    model = Prophet(interval_width=0.95)
                    unique_key_data = key_filter_data[key_filter_data['Key'] == i].copy()
                    if len(unique_key_data) > 0:
                        try:
                            model.fit(unique_key_data)
                            future = model.make_future_dataframe(periods=1, freq='W-Fri')
                            forecast = model.predict(future)
                            forecast = pd.merge(forecast, unique_key_data[['ds', 'y']], on='ds', how='left')
                            forecast = forecast.dropna(subset=['y'])
                            forecastAppend = forecast[['ds', 'yhat_lower', 'yhat_upper', 'y']].copy()
                            # st.write(i)

                            forecastAppend['Granularity'] = i
                            forecastAppend['Dimension'] = aggregateDimension
                            LW_R = forecastAppend.iloc[forecastAppend['ds'].idxmax()]
                            # print(LW_R)
                            # print(LW_R)
                            PATTERN_CHECKS_RESULTS = pd.concat([PATTERN_CHECKS_RESULTS, LW_R.to_frame().transpose()],ignore_index=True)
                            LW_R_DF = LW_R.to_frame().transpose()
                            OUTSIDE_FORECAST_COUNT = LW_R_DF[
                                (LW_R_DF['y'] < LW_R_DF['yhat_lower']) | (LW_R_DF['y'] > LW_R_DF['yhat_upper'])].shape[
                                0]
                            SEGMENTDROP_VAL = 'No' if OUTSIDE_FORECAST_COUNT == 0 else 'Yes'
                            print(f'Segment drops for {i}: {SEGMENTDROP_VAL}')
                        except Exception as e:
                            pass
                            print(e)

                # Plotting
                key_filter_data_52_weeks = key_filter_data[key_filter_data['ds'] > start_date_52_weeks_ago]
                key_filter_data_52_weeks['ds'] = pd.to_datetime(key_filter_data_52_weeks['ds'])

                grouped_data = key_filter_data_52_weeks.groupby(['ds', 'Key'])['y'].sum().unstack().reset_index()
                grouped_data = grouped_data.nlargest(5, 'ds').copy()
                grouped_data['ds'] = grouped_data['ds'].dt.strftime('%Y-%m-%d')
                # grouped_data[others_bucket] = grouped_data[columns_to_add].sum(axis=1)
                # grouped_data = grouped_data.drop(columns=columns_to_add)

                # Set the default width and height for all charts
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.set_option('deprecation.showfileUploaderEncoding', False)
                plt.xlim(0, 20)
                plt.figure(figsize=(10, 6))
                plt.xlim(0, 20)
                sns.set(style="whitegrid")

                # Calculate the cumulative sum of values for each group
                grouped_data = grouped_data.sort_values(by='ds')
                grouped_cumsum = grouped_data[unique_keys].cumsum(axis=1)

                # Plot stacked bars
                for i, t in enumerate(unique_keys):

                    if SEGMENTDROP_VAL == 'Yes':
                        grouped_data = grouped_data.drop(columns=['Others'], errors='ignore')
                        data_melted = grouped_data.melt(id_vars=['ds'], var_name='Category', value_name='Value')

                        # Get unique categories from the data
                        categories = set(data_melted['Category'])

                        # Create a dictionary to store category results
                        category_results = {}

                        # Iterate over each category
                        for category in categories:
                            category_data = data_melted[data_melted['Category'] == category]
                            filtered_data = category_data[category_data['Value'] > 0].tail(2)

                            if not filtered_data.empty and len(filtered_data) == 2:
                                # Extract the last two values
                                last_values = filtered_data['Value'].values
                                # Perform the subtraction
                                result = last_values[-1] - last_values[-2]
                                # Store the result in the dictionary
                                category_results[category] = result

                        # Initialize lists to store formatted results for increased and decreased values
                        formatted_results_increased = []
                        formatted_results_decreased = []

                        for category, value in category_results.items():
                            # Determine if the value is positive, negative, or zero
                            if value > 0:
                                upper_text = "has increased by"
                                formatted_result = f"Contribution percentage of {category} for the latest week {upper_text} {abs(value) * 100:.1f}% compared to the previous week."
                                formatted_results_increased.append(formatted_result)
                            elif value < 0:
                                upper_text = "has decreased by"
                                formatted_result = f"Contribution percentage of {category} for the latest week has experienced {abs(value) * 100:.1f}% {upper_text} compared to the previous week."
                                formatted_results_decreased.append(formatted_result)

                        # Combine all formatted results into a single string
                        result_string = "\n".join(formatted_results_increased + formatted_results_decreased)

                        # Print or use result_string as needed
                        chart = alt.Chart(data_melted).mark_bar(size=30).encode(
                            x=alt.X('ds:O', axis=alt.Axis(title='Date', labelAngle=-45), sort='ascending'),
                            y=alt.Y('sum(Value):Q', axis=alt.Axis(title='Contribution %', format='%')),
                            color='Category:N'
                        ).properties(width=800, height=400, title={'text': '     ', 'anchor': 'middle'})

                        # Define the last date and tooltip text
                        last_date = data_melted['ds'].max()
                        tooltip_text = (result_string)  # Change this to your actual tooltip text

                        # Create a red point chart
                        red_point = alt.Chart(
                            pd.DataFrame({'x': [last_date], 'y': [0.5], 'text': [tooltip_text]})).mark_circle(
                            color='red', size=100, filled=True
                        ).encode(
                            x='x:O',
                            y=alt.Y('y:Q', axis=alt.Axis(title='Contribution %', format='%')),
                            tooltip=alt.Tooltip('text:N', title='Text')
                        )

                        # Combine the bar chart and red point chart
                        final_chart = alt.layer(chart, red_point).configure_axis(grid=False)
                        container_pattern = st.container(border = True)
                        container_pattern_check.altair_chart(final_chart, use_container_width=True)


                        break






            def RUN_TIME_SERIES_ANALYSIS(filtered_data, selected_combination, LIST_OF_AGGREGATE_COLUMNS, METRIC_COLUMN,
                                         rca_text, metric_val):

                graphVal = 10
                global appendReal, graphNumber, container_Brand_volume

                filtered_data = filtered_data.loc[filtered_data['ds'] > '2023-01-01'].copy()
                sns_c = sns.color_palette(palette='deep')
                filtered_data.rename(columns={'Date': 'ds', METRIC_COLUMN: 'y'}, inplace=True)
                filtered_data_all = filtered_data.copy()

                filtered_data['ds'] = pd.to_datetime(filtered_data['ds'], errors='coerce', dayfirst=True,
                                                     format='%Y-%m-%d')
                # st.write('test2')
                max_date = filtered_data['ds'].max()
                start_date_52_weeks_ago = max_date - timedelta(weeks=216)
                filtered_data = filtered_data[filtered_data['ds'] > start_date_52_weeks_ago]
                filtered_data_historical = filtered_data_all[filtered_data_all['ds'] >= start_date_52_weeks_ago]

                # LIST_OF_AGGREGATE_COLUMNS.append('ds') if 'ds' not in LIST_OF_AGGREGATE_COLUMNS else None

                if 'ds' not in LIST_OF_AGGREGATE_COLUMNS:
                    LIST_OF_AGGREGATE_COLUMNS.append('ds')

                filtered_data_historical = filtered_data_historical.groupby(LIST_OF_AGGREGATE_COLUMNS).agg(
                    {'y': 'sum'}).reset_index()

                if len(filtered_data) == 0:
                    return 'Data not available for selected combination'

                # filtered_data['ds'] = filtered_data['ds'].dt.tz_localize(None)
                # excel_path = './Data/BeforeSum.xlsx'
                # filtered_data.to_excel(excel_path, index=False)

                filtered_data = filtered_data.groupby(LIST_OF_AGGREGATE_COLUMNS).agg({'y': 'sum'}).reset_index()

                df_holidays_manual = pd.read_csv('holidays_1.csv')
                df_holidays_manual = df_holidays_manual.dropna()

                model = Prophet(interval_width=0.95, holidays=df_holidays_manual)

                model.fit(filtered_data)
                future = model.make_future_dataframe(periods=1, freq='W-Fri')
                forecast = model.predict(future)
                forecast = pd.merge(forecast, filtered_data_historical[['ds', 'y']], on='ds', how='inner')
                forecast = forecast.sort_values(by='ds')

                forecast['y'] = pd.to_numeric(forecast['y'], errors='coerce')
                forecast['5%_increase'] = forecast['y'].shift(1) * 1.05
                forecast['5%_decrease'] = forecast['y'].shift(1) * 0.95

                result_string = ' | '.join(str(x) for x in selected_combination)
                # Split the result_string based on the delimiter '|'
                parts = result_string.split(' | ')

                # Save the two parts into separate lists
                Latest_Week = []
                Trend_Break = []

                if len(parts) == 2:
                    Latest_Week.append(parts[0])
                    Trend_Break.append(parts[1])
                else:
                    print("The result_string does not contain exactly two parts.")



                if graphNumber < graphVal:
                    ###############################################################################
                    # GRAPH CODE STARTS HERE
                    type_of_graphs = ['Manual Thresholds', 'ML Based Thresholds']

                    type_of_graphs = ['ML Based Thresholds']
                    for graph_type in type_of_graphs:
                        fig, ax = plt.subplots(figsize=(10, 8))

                        latest_data_point_index = -1
                        latest_data_point_color = 'r'
                        prior_data_points_color = 'gray'

                        forecast = forecast.merge(df_holidays_manual[['ds', 'holiday']], on='ds', how='left')

                        forecast['Legend'] = forecast.apply(lambda row: 'Holiday' if isinstance(row['holiday'], str)else ('Pass' if row['y'] >= row['yhat_lower'] and row['y'] <= row['yhat_upper'] else 'Fail'),axis=1)

                        holidays_data = forecast[forecast['Legend'] == 'Holiday']
                        pass_data = forecast[forecast['Legend'] == 'Pass']
                        fail_data = forecast[forecast['Legend'] == 'Fail']

                        holidays_data['rca_text'] = holidays_data['holiday'].copy()
                        fail_data['rca_text'] = ''

                        ax.plot(pass_data['ds'].values, pass_data['y'].values, 'o', color=prior_data_points_color,
                                markersize=2,
                                label='Pass Data')
                        ax.plot(fail_data['ds'].values, fail_data['y'].values, 'o', color=latest_data_point_color,
                                markersize=4,
                                label='Fail Data', zorder=5)
                        ax.plot(holidays_data['ds'].values, holidays_data['y'].values, 'o', color='blue', markersize=4,
                                label='Holidays', zorder=5)

                        fail_data = pd.concat([fail_data, holidays_data], ignore_index=True)

                        for index, row in fail_data.iterrows():
                            if not row['rca_text']:
                                filtered_data.rename(columns={'Date': 'ds', METRIC_COLUMN: 'y'}, inplace=True)
                                max_date = row['ds']
                                rca_input_df = filtered_data_all.copy()
                                if row['y'] > row['yhat_upper']:
                                    failureType = 'Upper'
                                else:
                                     failureType = 'Lower'



                                rca_input_df.rename(columns={'y': METRIC_COLUMN}, inplace=True)
                                
                                rca_text = custom_rca_for_each_week(max_date, rca_input_df, METRIC_COLUMN, metric_val,failureType)
                                fail_data.at[index, 'rca_text'] = rca_text

                        # convert the list into string
                        trend_break_string = ", ".join(Trend_Break)
                        latest_week_string = ", ".join(Latest_Week)

                        # set the color scale
                        color_scale = alt.Scale(domain=['Fail', 'Pass', 'Holiday'], range=['red', 'grey', 'blue'])

                        # Plot the data for rca
                        rac = alt.Chart(fail_data).mark_circle().encode(
                            x='ds:T',
                            y='y:Q',
                            color=alt.Color('Legend:N', scale=color_scale),
                            tooltip=[alt.Tooltip('rca_text:N', title="  ")]
                        ).properties(
                            width=800,
                            height=400,
                            title={
                                'text': f"{trend_break_string} Trend Break has been observed for {latest_week_string} in latest week",
                                'anchor': 'start'
                            }
                        )
                        # Plot the data for holiday
                        holiday = alt.Chart(holidays_data).mark_circle().encode(
                            x='ds:T',
                            y='y:Q',
                            color=alt.Color('Legend:N', scale=color_scale),
                            tooltip=[alt.Tooltip('holiday:N', title='Holiday')]
                        )

                        # Plot the data for pass
                        pass_chart = alt.Chart(pass_data).mark_circle().encode(
                            x='ds:T',
                            y='y:Q',
                            color=alt.Color('Legend:N', scale=color_scale),
                            tooltip=[alt.Tooltip('ds:T', title='Date'), alt.Tooltip('y:Q', title='Value')]
                        )

                        # Add ML-based bounds
                        ml_based_bounds = alt.Chart(forecast).mark_area(
                            color='blue',
                            opacity=0.2
                        ).encode(
                            x=alt.X('ds:T', title='Date'),
                            y=alt.Y('yhat_lower:Q', title='Metric'),
                            y2=alt.Y2('yhat_upper:Q', title='Volume')
                        )

                        #max_date_rca_text = fail_data.loc[fail_data['ds'] == fail_data['ds'].max(), 'rca_text'].iloc[0]

                        # max_date_rca_text = 'New test'
                        # ml_based_bounds_updated = ml_based_bounds + alt.Chart({'values': [{'x': 1, 'y': 1, 'text': max_date_rca_text}]}).mark_text(
                        #                                             align='right',
                        #                                             baseline='top',
                        #                                             dx=-10,  # Adjust this value to adjust the horizontal position
                        #                                             dy=10,   # Adjust this value to adjust the vertical position
                        #                                             fontSize=12,
                        #                                             fontWeight='bold',
                        #                                             color='black'
                        #                                         ).encode(
                        #                                             x=alt.value(1),  # Positioning the text at x=1 (right side)
                        #                                             y=alt.value(1),  # Positioning the text at y=1 (top)
                        #                                             text=alt.value(max_date_rca_text)
                        #                                         )

                        # Combine the fail data chart, ML-based bounds, holiday names, and pass data points
                        combined_chart = alt.layer(rac, ml_based_bounds, holiday, pass_chart).configure_axis(grid=False)

                        if graph_type == 'ML Based Thresholds':
                            ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                                            color='blue', alpha=0.2, label='ML Based Bounds')
                        elif graph_type == 'Manual Thresholds':
                            ax.fill_between(forecast['ds'], forecast['5%_decrease'], forecast['5%_increase'],
                                            color='red', alpha=0.2, label='Calc Interval')

                            # ax.plot(forecast['ds'], forecast['yhat'], color='grey', label='Trend')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Value')

                        lower_buffer = 100
                        buffer = 200

                        columns_to_check = ['yhat_upper', 'y']
                        max_value_specific_columns = forecast[columns_to_check].max().max()

                        ax.set_ylim(min(forecast['yhat_lower']) - lower_buffer, max_value_specific_columns + buffer)

                        # ax.legend()

                        legend = ax.legend(loc='lower right')

                        # Set legend font size
                        font_size = 5  # Change font size as needed
                        for text in legend.get_texts():
                            text.set_fontsize(font_size)

                        plt.title(str(result_string))
                    # gRaph code ends here
                    ################################################################################

                    graphNumber = graphNumber + 1

                count_outside_percent_interval = \
                    forecast[
                        (forecast['y'] < forecast['5%_decrease']) | (forecast['y'] > forecast['5%_increase'])].shape[0]
                count_outside_forecast = \
                    forecast[(forecast['y'] < forecast['yhat_lower']) | (forecast['y'] > forecast['yhat_upper'])].shape[
                        0]
                selected_row = forecast.loc[forecast['ds'].idxmax()]

                # append to new dataframe
                forecastAppend = forecast[['ds', 'yhat_lower', 'yhat_upper', 'y']].copy()

                forecastAppend['Granularity'] = result_string
                sr2 = forecastAppend.iloc[forecastAppend['ds'].idxmax()]

                appendReal = pd.concat([appendReal, sr2.to_frame().transpose()], ignore_index=True)

                if selected_row['yhat_lower'] <= selected_row['y'] <= selected_row['yhat_upper']:

                    latest_data_point_color = 'g'
                    print(f"Completed Time series Analysis for Combination - {selected_combination} with Result: ",
                          end="");
                    display(HTML("<font color='green'><b>PASS</b></font>"))


                else:
                    latest_data_point_color = 'r'
                    #         if graphNumber < graphVal:
                #             ax.text(0.03, 0.95, rca_text, fontsize=7, ha='left', va='top', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))

                #     if graphNumber < graphVal:
                #         ax.plot(forecast['ds'].iloc[latest_data_point_index], forecast['y'].iloc[latest_data_point_index], f'{latest_data_point_color}.', markersize=16,label='Latest Data')
                if selected_row['yhat_lower'] <= selected_row['y'] <= selected_row['yhat_upper']:
                    pass
                else:
                    container_Brand_volume.write(result_string)
                    container_Brand_volume.altair_chart(combined_chart, use_container_width=True)
                    # container_Brand_volume.write(rca_text)
                    # st.write(result_string)
                    # st.altair_chart(combined_chart, use_container_width=True)

            def unique_sorted_values_plus_ALL(array):
                unique = array.unique().tolist()
                unique.sort()
                return unique

            data_freshness_date = ''
            if selected_cell=="HCP_DATA":
                data_freshness_date = '2023-05-30'
            elif selected_cell=="US_RPT.D_RPT_SLS":
                data_freshness_date = '2024-02-02'
            THRESHOLD_DATE = '2024-01-12'
            graphNumber = 0
            appendReal = pd.DataFrame(columns=['ds', 'yhat_lower', 'yhat_upper', 'y', 'Granularity'])
            PATTERN_CHECKS_RESULTS = pd.DataFrame(columns=['ds', 'yhat_lower', 'yhat_upper', 'y', 'Granularity','Dimension'])
            CONFIG_DF = config_df
            CONFIG_DF = CONFIG_DF[(CONFIG_DF['ID'] == number)].copy()

            # Novartis_Brands = ['COSENTYX IV', 'COSENTYX', 'COSENTYX SC']
            #Novartis_Brands = ['Brand1', 'Brand2', 'Brand3']
            Novartis_Brands = brands_list

            def INITIATE_DATA_QUALITY(EXTRACTED_DATA_FROM_DB, LIST_OF_AGGREGATE_COLUMNS, DATE_COLUMN_IN_DATASET,
                                      METRIC_COLUMN, textDict):
                UNIQUE_COMBINATIONS = EXTRACTED_DATA_FROM_DB[LIST_OF_AGGREGATE_COLUMNS].drop_duplicates().apply(tuple,
                                                                                                                axis=1).tolist()
                # UNIQUE_COMBINATIONS = [('HS', 'COMPETITIVE', 'COSENTYX', 'MOTRX'),('PSO', 'COMPETITIVE', 'CIMZIA', 'MOTRX')]
                # UNIQUE_COMBINATIONS = [('COSENTYX', 'NBRX'),('COSENTYX SC', 'NBRX')]
                for CURRENT_COMBINATION in UNIQUE_COMBINATIONS:
                    # st.write(CURRENT_COMBINATION)
                    metric_val = CURRENT_COMBINATION[-1]
                    try:
                        rca_text = textDict[CURRENT_COMBINATION[-1]]
                    except Exception as e:
                        rca_text = ''
                    FILTER_DATA_FOR_EACH_COMBINATION(CURRENT_COMBINATION, LIST_OF_AGGREGATE_COLUMNS,
                                                     EXTRACTED_DATA_FROM_DB, METRIC_COLUMN, rca_text, metric_val)

            def INITIATE_DATA_QUALITY_PATTERN_CHECK(EXTRACTED_DATA_FROM_DB_FOR_DATA_PATTERN_CHECKS, AGGREGATE_COLUMNS,METRIC_COLUMN_NAME, VALUE_COLUMN_NAME, DATE_COLUMN_IN_DATASET,PIVOT_FLAG):

                if PIVOT_FLAG == "Yes":
                    for AGGREGATE_DIMENSION_COLUMN_NAME in AGGREGATE_COLUMNS:
                        METRIC_VALUE = None
                        PERFORM_DATA_PATTERN_CHECKS(METRIC_VALUE, AGGREGATE_DIMENSION_COLUMN_NAME,
                                                    DATE_COLUMN_IN_DATASET, METRIC_COLUMN_NAME, VALUE_COLUMN_NAME,
                                                    EXTRACTED_DATA_FROM_DB_FOR_DATA_PATTERN_CHECKS, PIVOT_FLAG)
                else:
                    UNIQUE_COMBINATIONS = EXTRACTED_DATA_FROM_DB_FOR_DATA_PATTERN_CHECKS[METRIC_COLUMN_NAME].drop_duplicates()

                    for CURRENT_COMBINATION in UNIQUE_COMBINATIONS:

                        FILTERED_DATA_FOR_CURRENT_METRIC = EXTRACTED_DATA_FROM_DB_FOR_DATA_PATTERN_CHECKS[(
                                    EXTRACTED_DATA_FROM_DB_FOR_DATA_PATTERN_CHECKS[
                                        METRIC_COLUMN_NAME] == CURRENT_COMBINATION)].copy()
                        if len(FILTERED_DATA_FOR_CURRENT_METRIC) > 0:
                            # AGGREGATE_COLUMNS = ['SPECIALTY']

                            for DIMENSION in AGGREGATE_COLUMNS:
                                METRIC_VALUE = CURRENT_COMBINATION
                                AGGREGATE_DIMENSION_COLUMN_NAME = DIMENSION
                                PERFORM_DATA_PATTERN_CHECKS(METRIC_VALUE, AGGREGATE_DIMENSION_COLUMN_NAME,DATE_COLUMN_IN_DATASET, METRIC_COLUMN_NAME,VALUE_COLUMN_NAME, FILTERED_DATA_FOR_CURRENT_METRIC,
                                                            PIVOT_FLAG)
                        else:
                            st.write(f"No records available for combination {CURRENT_COMBINATION}")

            def change_text(row):
                change_amount = abs(round(row['abs'], 1))
                if change_amount > 2:
                    if row['pct_recent'] > row['pct_history']:
                        return f"{row['Column Name']} {row['List_Value']} saw a {change_amount}% increase in its contribution to volume compared to the previous week (R8W)."
                    else:
                        return f"{row['Column Name']} {row['List_Value']} saw a {change_amount}% decrease in its contribution compared to the previous week (R8W)."
                else:
                    return "Could not identify the potential reason for anomaly"

            def process_config_row(row):
                global RCAdf, container_Brand_volume, container_pattern_check, freshnessDf, uniqueDf, notnullDf ,output_file
                global EXTRACTED_DATA_FROM_DB
                EXTRACTED_DATA_FROM_DB = pd.read_csv(selected_File_Name)
         
                TABLE_NAME = row['Table Name']
                FILTER_TO_BE_APPLIED = row['Filter']
                DATE_COLUMN_IN_DATASET = row['Date_Field']
                AGGREGATE_COLUMNS = row['Aggregate Columns']
                PIVOT_FLAG = row['Pivot Down']
                columns_for_freshness = row['Data Freshness'].split(';')
                columns_for_volume = row['Data Volume'].split(';')
                columns_for_not_nulls = row['Not Null'].split(';')
                columns_to_check_uniqueness = row['Uniqueness'].split(';')
                columns_for_RCA_analysis = row['RCA Analysis Columns'].split(';')
                AGGREGATE_COLUMNS_RAW = row['Dimension Patterns'].split(';')
                compund_key_columns_list = row['Table Level Uniqueness'].split(';')

                if PIVOT_FLAG == 'No':
                    METRIC_COLUMNS = row['Dimension Patterns Metric'].split(';')
                else:
                    METRIC_COLUMNS = row['Aggregate Metric'].split(';')

                metric_column_name = row['Metric Column Name']

                LIST_OF_AGGREGATE_COLUMNS = AGGREGATE_COLUMNS.split(';')
                # st.write(LIST_OF_AGGREGATE_COLUMNS)
                if len(str(FILTER_TO_BE_APPLIED)) > 3:
                    FILTER_FOR_QUERY = f"where {FILTER_TO_BE_APPLIED}"
                else:
                    FILTER_FOR_QUERY = ""

                st.write(f"Starting DQ Monitoring for {TABLE_NAME}")
                query = f"select * from {TABLE_NAME} {FILTER_FOR_QUERY}"

                # st.write(f"Fetched {len(EXTRACTED_DATA_FROM_DB)} records from Snowflake")

                try:
                    EXTRACTED_DATA_FROM_DB[DATE_COLUMN_IN_DATASET] = EXTRACTED_DATA_FROM_DB[
                        DATE_COLUMN_IN_DATASET].apply(lambda x: x.strftime('%Y-%m-%d'))
                except Exception as e:
                    pass

                # EXTRACTED_DATA_FROM_DB[DATE_COLUMN_IN_DATASET] = pd.to_datetime(EXTRACTED_DATA_FROM_DB[DATE_COLUMN_IN_DATASET], errors='coerce', dayfirst=True, format='%Y-%m-%d')

                EXTRACTED_DATA_FROM_DB[DATE_COLUMN_IN_DATASET] = pd.to_datetime(
                    EXTRACTED_DATA_FROM_DB[DATE_COLUMN_IN_DATASET], errors='coerce', dayfirst=True, format='%m/%d/%Y')
                EXTRACTED_DATA_FROM_DB[DATE_COLUMN_IN_DATASET] = pd.to_datetime(
                    EXTRACTED_DATA_FROM_DB[DATE_COLUMN_IN_DATASET], format='%d-%m-%Y', errors='coerce')

                new_column_names = {DATE_COLUMN_IN_DATASET: 'ds'}
                EXTRACTED_DATA_FROM_DB = EXTRACTED_DATA_FROM_DB.rename(columns=new_column_names)

                baseline = EXTRACTED_DATA_FROM_DB.isnull().sum().reset_index()
                baseline_obs = EXTRACTED_DATA_FROM_DB.shape[0]
                baseline.columns = ['Column', 'Null Count']


                    
                # Data profiling code start here
                ########################################################################
                # container_data_profiling = st.container(border=True)


                # bold_text = f"Data Profiling"
                # styled_text = f"<div style='font-weight: bold; color: black; font-size: 20px; margin-top: -10px;'>{bold_text}</div>"
                # container_data_profiling.markdown(styled_text, unsafe_allow_html=True)
                # container_data_profiling.write('    ')
                # container_data_profiling.write('    ')

                # #container_data_profiling.markdown("Data Profilling")
                # df = pd.read_csv(r"result.csv")

                # result_data = []
                # for col in df.columns:
                #     # Get the unique values and their counts in a column
                #     unique_values = df[col].value_counts().index.tolist()
                #     counts = df[col].value_counts().values.tolist()
                #     total_count = sum(counts)

                #     percentages = [(count / total_count) * 100 for count in counts]

                #     # Check if the number of unique values is greater than 50
                #     if total_count == 0:
                #         unique_values_str = 'No Values'
                #         percentages_str = 'No Values'
                #     else:
                #         percentages = [(count / total_count) * 100 for count in counts]

                #         # Check if the number of unique values is greater than 50
                #         if len(unique_values) > 50:
                #             unique_values = ['# of Values > 50']
                #             percentages = [100]

                #         # Create comma-separated string of unique values
                #         unique_values_str = '/'.join(unique_values)

                #         # Create comma-separated string of percentages
                #         percentages_str = ','.join([f'{percent:.2f}' for percent in percentages])

                #     # Append the results to the list
                #     result_data.append({'Column Name': col, 'Unique Values': unique_values_str,
                #                         'Percentage of Occurrence': percentages_str})

                # # Create DataFrame from the list of dictionaries
                # result_df = pd.DataFrame(result_data)

                # def off(x):
                #     e = []
                #     if 'No' not in x:
                #         for i in x.split(','):
                #             e.append(i)
                #         return e
                #     else:
                #         return x

                # result_df['Percentage_of_Occurrence'] = result_df['Percentage of Occurrence'].apply(lambda x: off(x))
                # result_df.to_excel('color.xlsx', index=False)

                # df = pd.read_excel(r"color.xlsx")

                # def parse_segment_values(segment_values_str):
                #     if segment_values_str == 'No Values':
                #         return None
                #     try:
                #         segment_values_list = ast.literal_eval(segment_values_str)
                #         return [float(val) for val in segment_values_list]
                #     except (ValueError, SyntaxError):
                #         return None

                # def print_colored_box_with_values(row, max_width=500):
                #     Percentage_of_Occurrence = parse_segment_values(row)
                #     if Percentage_of_Occurrence is None:
                #         return ''
                #     total = sum(Percentage_of_Occurrence)
                #     colors = ['#87CEEB', '#F0F8FF', '#F5F5F5', '#DCDCDC', '#F5FFFA', '#FAFAD2', '#E0FFFF', '#F0FFF0',
                #             '#FFE4E1',
                #             '#808000', '#800080', '#008080', '#c0c0c0', '#800000', '#808080']  # Adding more colors
                #     boxes = [
                #         '<div style="background-color:{}; width:{}px; display:inline-block; height:20px; position:relative;">'.format(
                #             colors[i % len(colors)], int(segment * max_width / total)) +
                #         '<span style="position:absolute; top:50%; left:50%; transform: translate(-50%, -50%); color:black; font-size:10px;">{}</span>'.format(
                #             '# of Values > 50' if Percentage_of_Occurrence == [100] else '{:.2f}'.format(segment)) +
                #         '</div>' for i, segment in enumerate(Percentage_of_Occurrence)]
                #     return ''.join(boxes)

                # df['% Of Values'] = df['Percentage_of_Occurrence'].apply(print_colored_box_with_values)
                # df.drop(columns=['Percentage_of_Occurrence', 'Percentage of Occurrence'], inplace=True)
                # df.reset_index(drop=True, inplace=True)
                # html_content = df.to_html(index= False, escape=False, classes='dataframe-wide')

                # styled_html = f"<div style='overflow-x:auto; max-height: 500px; width: 1400px;'>{html_content}</div>"
                # styled_html = f"<div style='overflow-x:auto; max-height: 500px; width: 1400px;'><style>th {{text-align: center;}}</style>{html_content}</div>"

                # # Use the styled_html with preserved HTML and custom styles

                # container_data_profiling.write(styled_html, unsafe_allow_html=True, hide_index=True)

                # Data profiling code ends here
                ########################################################################















                freshnessDf = pd.DataFrame(columns=['Dimension', 'Value', 'Status', 'Comment'])
                uniqueDf = pd.DataFrame(columns=['Dimension', 'Unique Check Result', 'Unique Check Comment.'])
                notnullDf = pd.DataFrame(columns=['Dimension', 'NOT NULL Check Result', 'NOT NULL Comment'])

                # volumeRowCountDf, notnullDf
                ##Check Function

                check_freshness(EXTRACTED_DATA_FROM_DB, data_freshness_date, columns_for_freshness)
                freshnessDf.to_csv('freshnessDf.csv', index=False)
                # check_data_volume_new(EXTRACTED_DATA_FROM_DB, data_freshness_date, columns_for_volume)
                output_file = "zero_check_results.csv"
                check_zeros(EXTRACTED_DATA_FROM_DB, output_file)


                # metric code start here

                
                Data_freshness = pd.read_csv("freshnessDf.csv")
                Data_Validation =pd.read_csv("Data_Validation.csv")
                Brand_Volume = pd.read_csv("Volume_Analysis.csv")
                Unknown_Unknown = pd.read_csv("PATTERN_CHECKS_RESULTS.csv")

                total_freshness = len(Data_freshness)
                correct_freshness = len(Data_freshness[Data_freshness["Status"] == "Pass"])

                total_validation = len(Data_Validation)
                correct_validation = len(Data_Validation[(Data_Validation["Status"] == "Pass")| (Data_Validation["Status"] == "Pass.")])

                total_Brand_volume = len(Brand_Volume)
                correct_Brand_volume = len(Brand_Volume[Brand_Volume["Status"] == "Pass"])

                total_Unknown = len(Unknown_Unknown)
                correct_Unknown = len(Unknown_Unknown[Unknown_Unknown["Status"] == "Pass"])

                # Metric Boxes
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    correctness_ratio = f"{correct_freshness}/{total_freshness}"
                    st.markdown(f"""
                                        <div style='text-align: center; background-color: #B3C9EC; color: black; padding: 5px; border-radius: 10px; max-width: 800px;'>
                                            <h8 style='margin: 0;'>Data Freshness</h8>
                                            <p style='font-size: 15px; margin: 0;'> <strong>{correctness_ratio}</strong></p>
                                        </div>
                                    """, unsafe_allow_html=True)

                with col2:
                    correctness_ratio = f"{correct_Brand_volume}/{total_Brand_volume}"
                    st.markdown(f"""
                                      <div style='text-align: center;background-color: #B3C9EC; color: black; padding: 5px; border-radius: 10px; max-width: 800px;'>
                                        <h8 style='margin: 0;'>Brand Volume</h8>
                                        <p style='font-size: 15px; margin: 0;'><strong>{correctness_ratio}</strong></p>
                                      </div>
                                  """, unsafe_allow_html=True)

                with col3:
                    correctness_ratio = f"{correct_Unknown}/{total_Unknown}"
                    st.markdown(f"""
                                     <div style='text-align: center; background-color: #B3C9EC; color: black; padding: 5px; border-radius: 10px; max-width: 800px;'>
                                       <h8 style='margin: 0;'>Table Anomalies</h8>
                                       <p style='font-size: 15px; margin: 0;'><strong>{correctness_ratio}</strong></p>
                                     </div>
                                """, unsafe_allow_html=True)

                with col4:
                    correctness_ratio = f"{correct_validation}/{total_validation}"
                    st.markdown(f"""
                                        <div style='text-align: center; background-color: #B3C9EC; color: black; padding: 5px; border-radius: 10px; max-width: 800px;'>
                                            <h8 style='margin: 0;'>Data Validation</h8>
                                            <p style='font-size: 15px; margin: 0;'><strong>{correctness_ratio}</strong></p>
                                        </div>
                                    """, unsafe_allow_html=True)

                with col5:
                    st.markdown("""
                                        <div style='text-align: center; background-color: #B3C9EC; color: black; padding: 5px; border-radius: 10px; max-width: 800px;'>
                                            <h8 style='margin: 0;'>Rule Based Checks</h3>
                                            <p style='font-size: 15px; margin: 0;'> <span style="font-weight:bold;">1/1</span></p>
                                        </div>
                                    """, unsafe_allow_html=True)

                st.write("\n\n")
                ##Check Function
                check_columns_new(EXTRACTED_DATA_FROM_DB, baseline, baseline_obs, data_freshness_date,columns_for_not_nulls, columns_to_check_uniqueness)
                uniqueDf.to_csv('uniqueDf.csv', index=False)
                notnullDf.to_csv('notnullDf.csv', index=False)
                container = st.container(border=True)
                container.markdown("""<style>.stApp {background-color: white; }</style>""", unsafe_allow_html=True)
                bold_text = f"Data Freshness/Recency Summary (WE - 02/02)"

                styled_text = f"<div style='font-weight: bold; color: black; font-size: 20px; margin-top: -10px;'>{bold_text}</div>"
                container.markdown(styled_text, unsafe_allow_html=True)

                container.write('    ')

                def highlight_pass_fail(val):
                    if val == 'Pass':
                        color = 'White'
                        text_color = 'Green'
                        symbol = '\u25cf'  # Green filled circle
                    elif val == 'Fail':
                        color = 'White'
                        text_color = 'Red'
                        symbol = '\u25cf'  # Red filled circle
                    else:
                        color = 'white'
                        text_color = 'black'
                        symbol = '\u25cf'  # Black filled circle
                    return f'background-color: {color}; color: {text_color}; text-align: center; padding: 2px 0px 2px 0px; content: "{symbol}"'

                config_df = pd.read_csv(r'./freshnessDf.csv')
                config_df.sort_values(by=['Comment'],inplace=True)

                config_df.rename(columns={'Comment': 'Comments'}, inplace=True)

                styled_df = config_df.style.applymap(highlight_pass_fail)
                container.dataframe(styled_df, width=1700, hide_index=True)

                # Render DataFrame with tooltips using HTML
                html_code = render_dataframe_with_tooltips(config_df)

                # Render HTML using Streamlit within a container

                #st.components.v1.html(html_code, height= 450)




                ##Table level Uniqueness Starts here 
                def is_compound_key(df, columns_list,date):
                    # st.write(columns_list)
                    # st.write(type(columns_list))
                    df = df.loc[df.ds == date]
                    df.to_csv('uniqueTable.csv')
                    is_unique = df.duplicated(columns_list).sum() == 0
                    if is_unique:
                        #container_data_uniqueness.write(f"The combination of columns forms a unique key for the latest week.\n {columns_list}")
                        listColumns = ', '.join(columns_list)
                        data = {'Dimension': listColumns,'Result': 'Pass','Comment': 'No Duplicates found'}
                        df = pd.DataFrame(data, index=[0])
                        styled_df = df.style.applymap(highlight_pass_fail)
                        container_data_uniqueness.dataframe(styled_df, width=1500, hide_index=True)

                        

                    else:
                        container_data_uniqueness.write("The combination of columns does not a unique key for the latest week.")
                        duplicated_records = df[df.duplicated(columns_list, keep=False)]
                        duplicated_records.to_csv('duplicated_records.csv')
                        #container_data_uniqueness.write(duplicated_records.head(10))

                container_data_uniqueness = st.container(border=True)
                container_data_uniqueness.markdown("""<style>.stApp {background-color: white; }</style>""", unsafe_allow_html=True)
                bold_text = f"Data Uniqueness Check (WE - 02/02)"
                styled_text = f"<div style='font-weight: bold; color: black; font-size: 20px; margin-top: -10px;'>{bold_text}</div>"
                container_data_uniqueness.markdown(styled_text, unsafe_allow_html=True)
                
                # Displaying the DataFrame
                #print(df)
                container_data_uniqueness.write('    ')
                is_compound_key(EXTRACTED_DATA_FROM_DB, compund_key_columns_list,data_freshness_date)
                
                ##Table level Uniqueness end here 




                container = st.container(border=True)
                container.markdown("""<style>.stApp {background-color: white; }</style>""", unsafe_allow_html=True)
                bold_text = f"Data Validations Summary (WE - 02/02)"
                styled_text = f"<div style='font-weight: bold; color: black; font-size: 20px; margin-top: -10px;'>{bold_text}</div>"
                container.markdown(styled_text, unsafe_allow_html=True)
                container.write('    ')

                def highlight_pass_fail(val):
                    if val == 'Pass':
                        color = 'White'
                        text_color = 'Green'
                        symbol = '\u25cf'  # Green filled circle
                    elif val == 'Fail':
                        color = 'White'
                        text_color = 'Red'
                        symbol = '\u25cf'  # Red filled circle
                    elif val == 'Pass.':
                        color = 'White'
                        text_color = '#ED7D31'
                        symbol = '\u25cf'  # Red filled circle
                    else:
                        color = 'white'
                        text_color = 'black'
                        symbol = '\u25cf'  # Black filled circle
                    #return f'background-color: {color}; color: {text_color}; text-align: center; padding: 2px 0px 2px 0px; content: "{symbol}"'**
                    return f'background-color: {color}; color: {text_color}; text-align: center; padding: 2px 0px 2px 0px; content: "{symbol}"'
                    #return **f'background-color: {color}; color: {text_color}; text-align: center; padding: 2px 0px 2px 0px; content: "{symbol}"'**

                # Read the CSV files
                df_unique = pd.read_csv(r'./uniqueDf.csv')
                #df_unique = df_unique.sort_values(by='Dimension', ascending=False)
                df_unique = df_unique.rename(columns={
                    "Unique Check Result": "Status",
                    "Unique Check Comment.": "Comments"})
                df_unique['Check'] = 'Unique'

                df_Null = pd.read_csv(r'./notnullDf.csv')
                #df_Null = df_Null.sort_values(by='Dimension', ascending=False)
                df_Null = df_Null.rename(columns={
                    "NOT NULL Check Result": "Status",
                    "NOT NULL Comment": "Comments"})
                df_Null['Check'] = 'Not_Null'
                df_Zero = pd.read_csv(r'./zero_check_results.csv')
                df_Zero['Check'] = 'Not_Zero'
                result = pd.concat([df_unique, df_Null,df_Zero], axis=0)





                result.reset_index(drop=True, inplace=True)
                result.to_csv("Data_Validation.csv")
                result = result.sort_values(by='Check', ascending=False)

                # Get unique conditions and add "ALL" option
                unique_conditions = result['Check'].unique().tolist()
                unique_conditions.insert(0, "ALL")
                
                # Select box for conditions with "ALL" option
                selected_condition = container.selectbox("Select Condition", unique_conditions)
                
                # Filter DataFrame based on selected condition
                if selected_condition == "ALL":
                    filtered_result = result
                else:
                    filtered_result = result[result['Check'] == selected_condition]
                styled_df = filtered_result.style.applymap(highlight_pass_fail)
                
                container.dataframe(styled_df, width=1700, hide_index=True)

                #html_code = render_dataframe_with_tooltips(result)

                # Render HTML using Streamlit within a container

                #st.components.v1.html(html_code, height=450)
                textDict = {'d1': 'dval'}

                container_Brand_volume = st.container(border=True)
                container_Brand_volume.markdown("""<style>.stApp {background-color: white; }</style>""",unsafe_allow_html=True)
                bold_text = f"DQ Summary of Brand Volume Checks (WE - 02/02)"
                styled_text = f"<div style='font-weight: bold; color: black; font-size: 20px; margin-top: -10px;'>{bold_text}</div>"
                container_Brand_volume.markdown(styled_text, unsafe_allow_html=True)
                container_Brand_volume.write('    ')

                def highlight_pass_fail(val):
                    if val == 'Pass':
                        color = 'White'
                        text_color = 'Green'
                        symbol = '\u25cf'  # Green filled circle
                    elif val == 'Fail':
                        color = 'White'
                        text_color = 'Red'
                        symbol = '\u25cf'  # Red filled circle
                    else:
                        color = 'white'
                        text_color = 'black'
                        symbol = '\u25cf'  # Black filled circle
                    return f'background-color: {color}; color: {text_color}; text-align: center; padding: 2px 0px 2px 0px; content: "{symbol}"'

                config_df = pd.read_csv(r'./Volume_Analysis.csv')

                columns_to_convert = ['Lower Threshold', 'Upper Threshold', 'Value']
                config_df[columns_to_convert] = config_df[columns_to_convert].apply(lambda x: pd.to_numeric(x, errors='coerce'))
                config_df[columns_to_convert] = config_df[columns_to_convert].round(0)
                config_df[columns_to_convert] = config_df[columns_to_convert].astype(str)

                # config_df[columns_to_convert] = config_df[columns_to_convert].apply(lambda x: x.rstrip('.0') if isinstance(x, str) and '.' in x else x)
                # config_df[columns_to_convert] = config_df[columns_to_convert].apply(lambda x: int(x.rstrip('.0')) if isinstance(x, str) and '.' in x else x)
                # config_df['Value'] = config_df['Value'].str.rstrip('.0').astype(int)
                # config_df['Lower Threshold'] = config_df['Lower Threshold'].str.rstrip('.0').astype(int)
                # config_df['Upper Threshold'] = config_df['Upper Threshold'].str.rstrip('.0').astype(int)
                # Replace empty strings with NaN
                
                config_df.drop(columns=['Date'], inplace=True)
                config_df = config_df[['Granularity', 'Status', 'Lower Threshold', 'Upper Threshold', 'Value']]
                config_df.rename(columns={'Value': 'Actuals'}, inplace=True)

                styled_df = config_df.style.applymap(highlight_pass_fail)

                container_Brand_volume.dataframe(styled_df, width=600, hide_index=True)

                #     #Volume checkss
                # st.write('checking volume')
                DATE_COLUMN_IN_DATASET = 'ds'

                if len(Novartis_Brands) > 0:
                    EXTRACTED_DATA_FROM_DB_NVS_BRANDS_FILTERED = EXTRACTED_DATA_FROM_DB[
                        EXTRACTED_DATA_FROM_DB['PRODUCT'].isin(Novartis_Brands)].copy()
                else:
                    EXTRACTED_DATA_FROM_DB_NVS_BRANDS_FILTERED = EXTRACTED_DATA_FROM_DB.copy()

                ##Check Function
                if PIVOT_FLAG == 'Yes':
                    AGGREGATE_METRICS = row['Aggregate Metric']
                    LIST_OF_AGGREGATE_METRIC_COLUMNS = AGGREGATE_METRICS.split(';')

                    for METRIC_COLUMN in LIST_OF_AGGREGATE_METRIC_COLUMNS:
                        INITIATE_DATA_QUALITY(EXTRACTED_DATA_FROM_DB_NVS_BRANDS_FILTERED, LIST_OF_AGGREGATE_COLUMNS,
                                              DATE_COLUMN_IN_DATASET, METRIC_COLUMN, textDict)
                else:
                    METRIC_COLUMN = metric_column_name

                    INITIATE_DATA_QUALITY(EXTRACTED_DATA_FROM_DB_NVS_BRANDS_FILTERED, LIST_OF_AGGREGATE_COLUMNS,
                                          DATE_COLUMN_IN_DATASET, METRIC_COLUMN, textDict)

                # ******************************************************##
                # Pattern Checks

                # ******************************************************##

                # bold_text = f"**DQ Summary of Data Pattern Checks**"
                # st.write(bold_text)

                # bold_text = f"DQ Summary of Data Pattern Checks (02/02)"
                # styled_text = f"<div style='border: 2px solid black; padding: 08px; font-weight: bold; color: black; font-size: 20px;'>{bold_text}</div>"
                # st.markdown(styled_text, unsafe_allow_html=True)
                # st.write('\n')

                container_pattern_check = st.container(border=True)
                container_pattern_check.markdown("""<style>.stApp {background-color: white; }</style>""",
                                                 unsafe_allow_html=True)
                bold_text = f"DQ Summary of Table Anomalies (02/02)"
                styled_text = f"<div style='font-weight: bold; color: black; font-size: 20px; margin-top: -10px;'>{bold_text}</div>"
                container_pattern_check.markdown(styled_text, unsafe_allow_html=True)
                container_pattern_check.write('    ')

                def highlight_pass_fail(val):
                    if val == 'Pass':
                        color = 'White'
                        text_color = 'Green'
                        symbol = '\u25cf'  # Green filled circle
                    elif val == 'Fail':
                        color = 'White'
                        text_color = 'Red'
                        symbol = '\u25cf'  # Red filled circle
                    else:
                        color = 'white'
                        text_color = 'black'
                        symbol = '\u25cf'  # Black filled circle
                    return f'background-color: {color}; color: {text_color}; text-align: center; padding: 2px 0px 2px 0px; content: "{symbol}"'

                config_df = pd.read_csv(r'./PATTERN_CHECKS_RESULTS.csv')
                config_df = config_df[config_df['Value'] > 0]
                config_df.drop(columns=['Date'], inplace=True)

                config_df['Unknown Unknown Checks'] = 'Segment Changes'
                #config_df['Dimension'] = 'Segment Changes'
                #config_df = config_df[['Unknown Unknown Checks', 'Granularity', 'Status', 'Value']]
                config_df = config_df[['Unknown Unknown Checks', 'Granularity', 'Status','Value']]
                # config_df[['Lower Threshold', 'Upper Threshold', 'Value']] = config_df[['Lower Threshold', 'Upper Threshold', 'Value']].round(2)

                #config_df = config_df.sort_values(by=['Status', 'Granularity'])
                config_df.index.name = 'MyIdx'
                config_df = config_df.sort_values(by=['Status', 'MyIdx'])

                columns_to_convert = ['Value']
                config_df[columns_to_convert] = config_df[columns_to_convert].apply(lambda x: pd.to_numeric(x, errors='coerce'))
                config_df[columns_to_convert] = config_df[columns_to_convert].round(3)


                config_df['Value'] = config_df['Value'] * 100
                config_df['Value'] = config_df['Value'].round(1)
                config_df['Value'] = config_df['Value'].astype(str) + '%'
                config_df.rename(columns={'Unknown Unknown Checks': 'Anomaly check'}, inplace=True)
                config_df["Comments"] = config_df.apply(lambda row:
                                                        f"{row['Granularity']} market contribution have been changed compared to last week"
                                                        if row["Status"] == "Fail"
                                                        else "",
                                                        axis=1)
                config_df.drop(columns=['Value'],inplace=True)
                styled_df = config_df.style.applymap(highlight_pass_fail)
                container_pattern_check.dataframe(styled_df, width=1700, hide_index=True)
                html_code = render_dataframe_with_tooltips(config_df)
                #st.components.v1.html(html_code, height=450)


                Novartis_Brands1 = []
                AGGREGATE_COLUMNS = []
                if len(Novartis_Brands1) > 0:
                    EXTRACTED_DATA_FROM_DB_NVS_BRANDS_FILTERED_Patterns = EXTRACTED_DATA_FROM_DB[
                        EXTRACTED_DATA_FROM_DB['PRODUCT'].isin(Novartis_Brands1)].copy()
                else:
                    EXTRACTED_DATA_FROM_DB_NVS_BRANDS_FILTERED_Patterns = EXTRACTED_DATA_FROM_DB.copy()

                for COLUMN_SET in AGGREGATE_COLUMNS_RAW:
                    if "," in COLUMN_SET:
                        AGGREGATE_COLUMNS.append(COLUMN_SET.split(','))
                    else:
                        AGGREGATE_COLUMNS.append(COLUMN_SET)

                    for METRIC_COLUMN_NAME in METRIC_COLUMNS:
                        if PIVOT_FLAG == 'No':
                            VALUE_COLUMN_NAME = row['Metric Column Name']
                        else:
                            VALUE_COLUMN_NAME = METRIC_COLUMN_NAME.strip()
                        # print(AGGREGATE_COLUMNS)

                INITIATE_DATA_QUALITY_PATTERN_CHECK(EXTRACTED_DATA_FROM_DB_NVS_BRANDS_FILTERED_Patterns,AGGREGATE_COLUMNS, METRIC_COLUMN_NAME, VALUE_COLUMN_NAME,DATE_COLUMN_IN_DATASET, PIVOT_FLAG)

                ####******************************************************##

            query = ''' select * from COM_US_PHARMA_COREPL.COREPL_ARD_NTL.NTL_COSENTYX_IMM_WKLY_NSOB_PROD where product in ('COSENTYX') and metric in ('NBRX') and  week >'2023-01-01' and "PRODUCT" not in ('ABRILADA') and "WEEK" <= '2024-02-02' ;  '''
            DATE_COLUMN_IN_DATASET_1 = 'Test'
            # EXTRACTED_DATA_FROM_DB = get_data_from_snowflake(query, DATE_COLUMN_IN_DATASET_1)

            CONFIG_DF.apply(process_config_row, axis=1)

            appendReal['yhat_lower'] = pd.to_numeric(appendReal['yhat_lower'], errors='coerce')
            appendReal['yhat_upper'] = pd.to_numeric(appendReal['yhat_upper'], errors='coerce')
            appendReal['Status'] = appendReal.apply(lambda row: 'Pass' if row['y'] >= row['yhat_lower'] and row['y'] <= row['yhat_upper'] else 'Fail',axis=1)

            new_column_names = {'ds': 'Date', 'yhat_lower': 'Lower Threshold', 'yhat_upper': 'Upper Threshold','y': 'Value'}
            appendReal = appendReal.rename(columns=new_column_names)
            appendReal.to_csv('Volume_Analysis.csv', index=False)

            PATTERN_CHECKS_RESULTS['Status'] = PATTERN_CHECKS_RESULTS.apply(lambda row: 'Pass' if row['y'] >= row['yhat_lower'] and row['y'] <= row['yhat_upper'] else 'Fail',axis=1)
            PATTERN_CHECKS_RENAME_COLUMNS = {'ds': 'Date', 'yhat_lower': 'Lower Threshold','yhat_upper': 'Upper Threshold', 'y': 'Value'}
            PATTERN_CHECKS_RESULTS = PATTERN_CHECKS_RESULTS.rename(columns=PATTERN_CHECKS_RENAME_COLUMNS)
            PATTERN_CHECKS_RESULTS.to_csv('PATTERN_CHECKS_RESULTS.csv')





            

if __name__ == "__main__":
    main()
