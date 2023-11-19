import streamlit as st
import pandas as pd
import numpy as np
from scorecard_pipeline import ScorecardPipeline
import scorecardpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature(feature, data):
    fig, ax1 = plt.subplots(figsize=(10,6))
    color = 'tab:green'
    
    ax1.set_title(feature, fontsize=16)
    ax1.set_xlabel('Bin', fontsize=16)
    ax1.set_ylabel('Size', fontsize=16, color=color)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15, horizontalalignment='right')

    sns.barplot(x='bin', y='size', data=data, palette='summer', ax=ax1)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('event_rate', fontsize=16, color=color)
    sns.lineplot(x='bin', y='event_rate', data=data, sort=False, color=color, ax=ax2)

    return fig

def analyse_score_by_outcome(preds_df, score_bin_type='percentile', score_bin_increment=10, sort_descending=False):
        score_df = preds_df.copy()
        score_df['count'] = 1
        if score_bin_type == 'percentile':
            score_df['score_bin'] = pd.qcut(np.round(score_df['_model_score_'],0).astype('int64'), 10).astype(str)
        else:
            score_df['score_bin'] = pd.cut(np.round(score_df['_model_score_'],0).astype('int64'), bins=range(0, 1000, score_bin_increment)).astype(str)

        score_df = score_df.groupby(['score_bin']).agg({
            'count':'sum', 
            'target':'sum',
            'probability':'mean'
            }).reset_index()
        if sort_descending:
            score_df = score_df.iloc[::-1]

        score_df.rename(columns={'probability':'event_mean_probability'}, inplace=True)
        score_df['relative_count'] = score_df['count'] / score_df['count'].sum()
        score_df['event_count'] = score_df['target']
        del score_df['target']
        score_df['non_event_count'] = score_df['count'] - score_df['event_count']
        score_df['event_rate'] = score_df['event_count'] / score_df['count']
        score_df['relative_cumulative_count'] = score_df['count'].cumsum() / score_df['count'].sum()
        score_df['relative_cumulative_event_count'] = score_df['event_count'].cumsum() / score_df['event_count'].sum()
        score_df['relative_cumulative_non_event_count'] = score_df['non_event_count'].cumsum() / score_df['non_event_count'].sum()
        return score_df

st.set_page_config(page_title="Scorecard creator", layout="wide")
st.title('Scorecard creator')
pd.set_option('display.max_columns', None)
is_done = False
error = False
create_scorecard = None

with st.sidebar:
    uploaded_file = st.file_uploader("Upload your data", type="csv")

    if uploaded_file is None:
        use_sample_data = True
        df = pd.read_csv('data/sample_data.csv')
        columns_list = list(df.columns)
        target_index = columns_list.index('approved') if 'approved' in columns_list else 0
        id_index = columns_list.index('id') if 'id' in columns_list else 0

    else:
        use_sample_data = False
        df = pd.read_csv(uploaded_file)
        columns_list = ['Not Selected'] + list(df.columns)
        target_index = 0
        id_index = 0
      
    target_col = st.selectbox("Select Target Column", columns_list, index=target_index)
    if target_col != 'Not Selected':
        df[target_col] = df[target_col].astype('int64')
        if sorted(df[target_col].unique()) != [0, 1]:
            error = True
            st.error("Error: Target column must contain exactly two unique values: 0 and 1")
        else:
            error = False
    id_col = st.selectbox("Select ID Column", columns_list, index=id_index)

    with st.expander("Pipeline options"):
        test_set_size = st.slider("Test size: Percentage of data allocated to test the model against", 0, 100, 20, 5) / 100
        min_bin_pct = st.number_input("Min bin percentage: The minimum percentage of data required to be within a bin", min_value=0.0, max_value=10.0, value=2.5, step=0.5) / 100
        correlation_cutoff = st.slider("Correlation cutoff: The minimum correlation allowed between features", 30, 100, 65, 5) / 100
        max_features = st.number_input("Max features: The number of features to train the model on based on predictiveness.", min_value=2, max_value=100, value=20, step=1)
        ref_odd = st.slider("Reference odds", 5, 50, 15, 5)
        ref_score = st.slider("Reference score", 500, 700, 660, 5)
        pdo = st.slider("Points to double the odds", 10, 50, 20, 5)

    with st.expander("Score analysis options"):
        score_bin_type = st.selectbox("Select score bin type", ['Percentile', 'Fixed increments'], index=0)
        sort_descending = st.checkbox("Analyse score bands in descending order", value=False)
        score_bin_increment = st.number_input("Select score bin increment", min_value=1, max_value=100, value=10, step=1)

    if not error:
        create_scorecard = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #ff4b4b;
            color:white;
        }
        </style>""",
        unsafe_allow_html=True)
        create_scorecard = st.button("Create Scorecard")

     st.markdown("""
        ## Contact Information
        If you have any questions or need further assistance, please feel free to reach out to me at [xanderhorn@hotmail.com](mailto:xanderhorn@hotmail.com).
        """)

if use_sample_data == True:
    st.warning("🚨 Using sample data. Upload your own data in the sidebar to create your own scorecard.")

with st.expander("Help"):

    st.write("""
    * **Target column**: Your target should be the data point you are trying to predict. A scorecard can only predict binary outcomes, so your target column should only contain two unique values. 
    The first unique value will be considered the non-event outcome (0), and the second unique value will be considered the event outcome (1). 
    If you are trying to predict who will default on their loan and the higher the score the less risky the individual, your **event** will be cases where a person did not default on their loan and your **non-event** default cases.
    Your target cannot contain any missing values.

    * **Reference Odds**: This is the odds of the event happening at the reference score. For example, if the reference odds are 50, it means that at the reference score, the odds of the event happening are 50 to 1.

    * **Reference Score**: This is a score that you choose as a reference point. The reference odds are the odds of the event happening at this score. The reference score is often chosen to be a round number like 500 or 600.

    * **Points to Double the Odds (PDO)**: This is the number of points that need to be added to the score to double the odds of the event. For example, if the PDO is 20, it means that adding 20 points to the score doubles the odds of the event.

    * **Scorecard development**: A scorecard is a simple model that allows you to assign points to predictor variables in order to calculate a score that predicts a certain outcome. The development of a scorecard involves selecting relevant predictor variables, determining the number of points for each variable, and validating the scorecard.

    * **Modeling**: In the context of scorecard development, modeling involves using statistical techniques to determine the relationship between the predictor variables and the target variable. This can involve techniques like logistic regression, decision trees, or machine learning algorithms.

    * **Predictor variables**: These are the variables that you use to predict the target variable. In a scorecard, each predictor variable is assigned a number of points based on its value. The total score is then calculated by adding up the points for each predictor variable.

    * **Points**: In a scorecard, points are assigned to the values of predictor variables. The number of points is determined based on the relationship between the predictor variable and the target variable. The total score is calculated by adding up the points for each predictor variable.

    * **Score**: The total score is calculated by adding up the points for each predictor variable. This score is used to predict the target variable. For example, in a credit scorecard, a higher score might indicate a lower risk of default.

    * **Validation**: After a scorecard has been developed, it needs to be validated to ensure that it accurately predicts the target variable. This can involve techniques like cross-validation or out-of-time validation.
    
    * **Target Distribution**: This is the distribution of the values in the target column. It shows how many times each value appears in the target column expressed as a percentage.

    * **ROC AUC (Receiver Operating Characteristic Area Under the Curve)**: This is a measure of how well a model can distinguish between different classes. The closer the ROC AUC is to 1, the better the model is. A value of 0.5 means the model is as good as a random guess. 

    * **Kolmogorov-Smirnov**: This is a statistical test that can be used to compare the distribution of scores for events and non-events. The higher the KS statistic, the better the model is at separating events from non-events.

    * **Gini Coefficient**: This is a measure of inequality among values of a frequency distribution (for example, levels of income). A Gini coefficient of zero expresses perfect equality, where all values are the same (for example, where everyone has the same income). A Gini coefficient of one (or 100%) expresses maximal inequality among values.

    * **Probability Distribution**: Shows how the probabilities of an event are distributed.
             
    * **Monotonic Weights of Evidence (WoE) Binning**: This is a technique used in scorecard development to transform a continuous predictor variable into a set of categories, each of which is associated with a Weight of Evidence (WoE) value. The WoE values are calculated in such a way that they increase or decrease monotonically with the predictor variable. This ensures that the relationship between the predictor variable and the target variable is captured in a way that makes sense for scorecard development.
             
    * **Scorecard Table**: Tabulates according to some score grouping, the number of data points, the event and event rate residing in each score grouping. Useful for analysing the scorecard and determining a cutoff to best meet business needs.

    * **Event Rate**: This is the proportion of the target population that experiences the event (e.g., default on a loan). In scorecard terms, this is often referred to as the 'bad' rate.         
    """)

with st.expander("Data overview"):
    if df.shape[0] > 0:
        st.write("Data preview")
        st.write(df.head())
        if target_col != 'Not Selected':
            target_distribution = pd.DataFrame(df[target_col].value_counts() / df.shape[0]).reset_index()
            target_distribution.columns = ['target', 'percentage_of_data']
            target_distribution['target'] = np.where(target_distribution['target'] == 0, 'Non-event', 'Event')
            st.write("Target distribution")
            st.write(target_distribution)

if create_scorecard and df.shape[0] > 0 and not error:
    if target_col != 'Not Selected':
        if sorted(df[target_col].unique()) != [0, 1]:
            st.error("Error: Target column must contain exactly two unique values: 0 and 1")
        else:
            with st.spinner('Processing... Please wait.'):
                pipeline = ScorecardPipeline(min_bin_pct=min_bin_pct, test_set_size=test_set_size, correlation_cutoff=correlation_cutoff, ref_odds=ref_odd, ref_score=ref_score, pdo=pdo, max_features=max_features)
                pipeline.fit(data=df, y=target_col, id_col=id_col)

                preds = pipeline.transform(df, pipeline.scorecard)
                p_train = preds.iloc[pipeline.train_index]['probability']
                p_test = preds.iloc[pipeline.test_index]['probability']
                train_perf_plot = sc.perf_eva(df.iloc[pipeline.train_index][target_col], preds.iloc[pipeline.train_index]['probability'], title="Training Set")['pic']
                test_perf_plot = sc.perf_eva( df.iloc[pipeline.test_index][target_col], preds.iloc[pipeline.test_index]['probability'], title="Testing Set")['pic']
                
                is_done = True  

with st.expander("Model Results"):
    if is_done:
        
        st.pyplot(train_perf_plot)
        st.pyplot(test_perf_plot)
        st.write(pipeline.performance)

        fig, ax = plt.subplots(figsize=(10,6))
        sns.kdeplot(p_train, shade=True, color="r", ax=ax, label="Train")
        sns.kdeplot(p_test, shade=True, color="b", ax=ax, label="Test").set(title='Probability distributions')
        ax.legend()
        st.pyplot(fig)

with st.expander("Model Features"):
    if is_done:

        for feature in pipeline.model.feature_names_in_:
            tmp_df = pipeline.woe_tables.loc[pipeline.woe_tables['feature'] == feature].copy()
            tmp_df['event_rate'].fillna(0, inplace=True)
            st.pyplot(plot_feature(feature, tmp_df))

with st.expander("Scorecard Results"):
    if is_done:
        fig, ax = plt.subplots(figsize=(10,6))
        mean_score = round(preds['_model_score_'].mean(),0)
        ax.hist(preds['_model_score_'],
            bins=150,
            edgecolor='white',
            color = '#317DC2',
            linewidth=1.2)

        ax.set_title(f'Scorecard Distribution, mean score = {mean_score}', fontweight="bold", fontsize=14)
        ax.axvline(preds['_model_score_'].mean(), color='k', linestyle='dashed', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10,6))
        ax = plt.scatter(x='_model_score_',
                    y='probability',
                    data=preds,
                    color='#317DC2')

        ax = plt.title('Scores by Probability', fontweight="bold", fontsize=14)
        ax = plt.xlabel('Score')
        ax = plt.ylabel('Probability (Bad)')
        ax = plt.yticks(np.arange(0, 1, 0.05));
        st.pyplot(fig)

        score_by_outcome = analyse_score_by_outcome(preds, score_bin_type=score_bin_type, score_bin_increment=score_bin_increment, sort_descending=sort_descending)
        st.write(score_by_outcome.style.background_gradient(subset=['relative_cumulative_non_event_count',"relative_cumulative_event_count"], cmap="Spectral", vmin=-1, vmax=1))

with st.expander("Scorecard"):
    if is_done:
        out_scorecard = pipeline.scorecard.copy()
        out_scorecard.drop(columns=['woe','coefficient'], axis=1, inplace=True)
        st.write(out_scorecard)

        csv = out_scorecard.to_csv(index=False)
        st.download_button(
            label="Download scorecard as CSV",
            data=csv,
            file_name="scorecard.csv",
            mime="text/csv",
        )
