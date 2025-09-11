import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageTemplate, Frame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics import renderPDF
import io

# Set page configuration
st.set_page_config(page_title="Loan Approval Prediction App", layout="wide")

# Custom CSS for improved UI
st.markdown("""
    <style>
    .main {background-color: #000000;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stSelectbox, .stSlider, .stNumberInput, .stTextInput {background-color: black; padding: 10px; border-radius: 5px;}
    .stTabs {background-color: black; padding: 10px; border-radius: 5px;}
    h1 {color: #ffffff;}
    h2, h3 {color: #ffffff;}
    .stMetric {background-color: black; border-radius: 5px; padding: 10px;}
    </style>
""", unsafe_allow_html=True)

# Load the dataset from GitHub URL
@st.cache_data
def load_data():
    try:
        url = "https://raw.githubusercontent.com/sreenat200/loan_approval/bd4a116c1456c8ad7ef29a6b62eb20ba5f3efa61/loan_data.csv"
        data = pd.read_csv(url, delimiter=',', skipinitialspace=True, on_bad_lines='skip')
        return data
    except Exception as e:
        st.error(f"Error loading file from GitHub: {e}")
        st.write("Please upload a local loan_data.csv file as a fallback.")
        uploaded_file = st.file_uploader("Upload loan_data.csv", type=["csv"])
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file, delimiter=',', skipinitialspace=True, on_bad_lines='skip')
                return data
            except Exception as e2:
                st.error(f"Error loading uploaded file: {e2}")
        return None

# Preprocess the data
def preprocess_data(data):
    if data is None:
        return None, None, None, None
    # All features by default
    selected_features = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
                        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
    categorical_cols = ['person_gender', 'person_education', 'person_home_ownership', 
                        'loan_intent', 'previous_loan_defaults_on_file']
    data_encoded = pd.get_dummies(data, columns=[col for col in categorical_cols if col in data.columns], drop_first=True)
    
    # Select features (numeric + encoded categorical)
    available_features = [col for col in data_encoded.columns if col != 'loan_status']
    X = data_encoded[available_features]
    y = data['loan_status']
    
    # Handle missing values
    X = X.fillna(X.median())
    y = y.fillna(y.mode()[0])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, available_features

# Train the model
@st.cache_resource
def train_model(X, y, algorithm):
    if X is None or y is None:
        return None, None, None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if algorithm == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    elif algorithm == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif algorithm == "XGBoost":
        model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if algorithm in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        feature_importance = model.feature_importances_
    else:
        feature_importance = np.abs(model.coef_[0]) / np.abs(model.coef_[0]).sum()
    
    return model, accuracy, precision, recall, f1, feature_importance, X_test, y_test, y_pred

# Generate stylish PDF report with premium look and border
def generate_pdf_report(input_data, prediction, probability, accuracy, precision, recall, f1, feature_importance, features, algorithm, customer_name="", customer_phone=""):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.5*inch, leftMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    # Custom styles for premium look
    title_style = ParagraphStyle(
        name='TitleStyle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=16, textColor=colors.navy,
        spaceAfter=20, alignment=1
    )
    heading_style = ParagraphStyle(
        name='HeadingStyle', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, textColor=colors.darkblue,
        spaceBefore=10, spaceAfter=10
    )
    normal_style = ParagraphStyle(
        name='NormalStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=10, textColor=colors.black,
        spaceAfter=8
    )
    
    elements = []
    
    # Add border to the page
    def add_page_border(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.darkblue)
        canvas.setLineWidth(2)
        canvas.rect(0.3*inch, 0.3*inch, letter[0]-0.6*inch, letter[1]-0.6*inch)
        canvas.restoreState()
    
    doc.addPageTemplates([PageTemplate(id='Bordered', frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height)], onPage=add_page_border)])
    
    # Header
    elements.append(Paragraph(f"Loan Approval Prediction Report ({algorithm})", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Customer Info
    if customer_name or customer_phone:
        elements.append(Paragraph("Customer Information", heading_style))
        customer_data = [["Field", "Value"]]
        if customer_name:
            customer_data.append(["Name", customer_name])
        if customer_phone:
            customer_data.append(["Phone Number", customer_phone])
        customer_table = Table(customer_data, colWidths=[2*inch, 4*inch])
        customer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BOX', (0, 0), (-1, -1), 1, colors.darkblue)
        ]))
        elements.append(customer_table)
        elements.append(Spacer(1, 0.2*inch))
    
    # Input Features
    elements.append(Paragraph("Input Features", heading_style))
    data = [["Feature", "Value"]]
    for feature, value in input_data.items():
        data.append([feature.replace('_', ' ').title(), f"{value:.2f}" if isinstance(value, (int, float)) else str(value)])
    table = Table(data, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.darkblue)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Prediction
    elements.append(Paragraph(f"Prediction: {'Rejected' if prediction == 1 else 'Approved'}", heading_style))
    elements.append(Paragraph(f"Probability of Rejection: {probability:.2%}", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Model Performance
    elements.append(Paragraph("Model Performance", heading_style))
    elements.append(Paragraph(f"Algorithm: {algorithm}", normal_style))
    elements.append(Paragraph(f"Accuracy: {accuracy:.4f}", normal_style))
    elements.append(Paragraph(f"Precision: {precision:.4f}", normal_style))
    elements.append(Paragraph(f"Recall: {recall:.4f}", normal_style))
    elements.append(Paragraph(f"F1-Score: {f1:.4f}", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Feature Importance
    elements.append(Paragraph("Feature Importance", heading_style))
    importance_data = [["Feature", "Importance"]]
    for feature, importance in zip(features, feature_importance):
        importance_data.append([feature.replace('_', ' ').title(), f"{importance:.4f}"])
    importance_table = Table(importance_data, colWidths=[2*inch, 4*inch])
    importance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.darkblue)
    ]))
    elements.append(importance_table)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Main app
def main():
    st.title("💰 Loan Approval Prediction App")
    st.markdown("Predict loan approval outcomes with live updates and interactive visualizations.")

    # Load and preprocess data
    data = load_data()
    if data is None:
        return

    # Sidebar for user input
    st.sidebar.header("📝 Input Details")
    customer_name = st.sidebar.text_input("Customer Name (Optional)", key="customer_name")
    customer_phone = st.sidebar.text_input("Phone Number (Optional)", key="customer_phone")
    algorithm = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "Logistic Regression", "Gradient Boosting", "XGBoost"])
    
    input_data = {}
    numeric_features = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
                        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
    categorical_features = {
        'person_gender': ['male', 'female'],
        'person_education': ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'],
        'person_home_ownership': ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
        'loan_intent': ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
        'previous_loan_defaults_on_file': ['Yes', 'No']
    }
    
    for feature in numeric_features:
        try:
            if feature in ['person_age', 'person_emp_exp', 'cb_person_cred_hist_length']:
                input_data[feature] = st.sidebar.slider(
                    feature.replace('_', ' ').title(),
                    min_value=int(data[feature].min()),
                    max_value=int(data[feature].max()),
                    value=int(data[feature].median()),
                    key=f"{feature}_input"
                )
            else:
                input_data[feature] = st.sidebar.number_input(
                    feature.replace('_', ' ').title(),
                    min_value=float(data[feature].min()),
                    max_value=float(data[feature].max()),
                    value=float(data[feature].median()),
                    step=0.1,
                    key=f"{feature}_input"
                )
        except ValueError as e:
            st.error(f"Error processing {feature}: {e}. Please check the dataset for non-numeric values.")
            return
    
    for feature, options in categorical_features.items():
        input_data[feature] = st.sidebar.selectbox(
            feature.replace('_', ' ').title(),
            options=options,
            index=options.index(data[feature].mode()[0]),
            key=f"{feature}_input"
        )

    # Tabs for prediction and visualizations
    tab1, tab2 = st.tabs(["Prediction", "Visualizations"])

    with tab1:
        st.header("Live Prediction")
        st.write("Enter borrower details to predict loan approval status.")
        # Train model with all features
        X, y, scaler, feature_names = preprocess_data(data)
        if X is None:
            return
        model, accuracy, precision, recall, f1, feature_importance, X_test, y_test, y_pred = train_model(X, y, algorithm)
        if model is None:
            return
        
        # Live prediction based on sidebar inputs
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df, columns=categorical_features.keys(), drop_first=True)
        input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
        input_scaled = scaler.transform(input_encoded)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.success(f"### Prediction: **{'Rejected' if prediction == 1 else 'Approved'}**")
        st.write(f"Probability of Rejection: **{probability:.2%}**")

        pdf_buffer = generate_pdf_report(
            input_data,
            prediction,
            probability,
            accuracy,
            precision,
            recall,
            f1,
            feature_importance,
            feature_names,
            algorithm,
            customer_name,
            customer_phone
        )
        st.download_button(
            label="📥 Download Prediction Report (PDF)",
            data=pdf_buffer,
            file_name=f"loan_approval_prediction_{algorithm.replace(' ', '_').lower()}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        st.header("📈 Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Algorithm", algorithm)
        col2.metric("Accuracy", f"{accuracy:.4f}")
        col3.metric("Precision", f"{precision:.4f}")
        col4.metric("Recall", f"{recall:.4f}")
        col5 = st.columns(1)[0]
        col5.metric("F1-Score", f"{f1:.4f}")

    with tab2:
        st.header("📊 Data Visualizations")
        st.write("Explore model performance and data insights.")
        
        graph_type = st.selectbox(
            "Select Visualization Type",
            options=["Confusion Matrix", "ROC Curve", "Feature Importance", "Scatter Plot"],
            key="graph_type"
        )
        
        if graph_type == "Scatter Plot":
            st.subheader("Select Features for Scatter Plot")
            x_feature = st.selectbox("X-axis Feature", options=numeric_features, key="x_feature")
            y_feature = st.selectbox("Y-axis Feature", options=numeric_features, key="y_feature")
            color_feature = st.selectbox("Color by", options=['loan_status'] + numeric_features, key="color_feature")
            size_feature = st.selectbox("Size by", options=["None"] + numeric_features, key="size_feature")
        
        if graph_type == "Confusion Matrix":
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Approved', 'Rejected'],
                y=['Approved', 'Rejected'],
                text_auto=True,
                title="Confusion Matrix"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        elif graph_type == "ROC Curve":
            st.subheader("ROC Curve")
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig_roc.update_layout(
                title="Receiver Operating Characteristic (ROC) Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                showlegend=True
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        elif graph_type == "Feature Importance":
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=True)
            fig_importance = px.bar(
                importance_df, x='Importance', y='Feature', orientation='h',
                title=f"Feature Importance ({algorithm})",
                color='Importance', color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_importance, use_container_width=True)

        elif graph_type == "Scatter Plot":
            st.subheader(f"{x_feature.replace('_', ' ').title()} vs {y_feature.replace('_', ' ').title()}")
            fig_scatter = px.scatter(
                data,
                x=x_feature,
                y=y_feature,
                color=color_feature,
                size=size_feature if size_feature != "None" else None,
                title=f"{x_feature.replace('_', ' ').title()} vs {y_feature.replace('_', ' ').title()} (Colored by {color_feature.replace('_', ' ').title()})",
                labels={x_feature: x_feature.replace('_', ' ').title(), y_feature: y_feature.replace('_', ' ').title(), color_feature: color_feature.replace('_', ' ').title()}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

if __name__ == "__main__":
    main()
