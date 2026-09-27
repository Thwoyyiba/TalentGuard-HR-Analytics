from flask import Flask, render_template, request

import joblib

import pandas as pd

import numpy as np



# ============================================================

# FLASK APP

# ============================================================



app = Flask(__name__)





# ============================================================

# LOAD SAVED FILES

# ============================================================



model = joblib.load(

    "Random Forest Classifier.pkl"

)



scaler = joblib.load(

    "X_Scaled.pkl"

)



feature_names = joblib.load(

    "feature_names.pkl"

)



recommendation_mapping = joblib.load(

    "recommendation_mapping.pkl"

)



# SHAP explainer

explainer = joblib.load(

    "shap_explainer.pkl"

)





# ============================================================

# HOME PAGE

# ============================================================



@app.route("/")

def home():



    return render_template(

        "index.html"

    )





# ============================================================

# PREDICTION

# ============================================================



@app.route(

    "/predict",

    methods=["POST"]

)

def predict():



    try:



        # ====================================================

        # GET FORM INPUTS

        # ====================================================



        Age = float(request.form["Age"])

        Gender = request.form["Gender"]

        Marital_Status = request.form["Marital_Status"]

        Department = request.form["Department"]

        Job_Role = request.form["Job_Role"]

        Job_Level = int(request.form["Job_Level"])



        Monthly_Income = float(

            request.form["Monthly_Income"]

        )



        Hourly_Rate = float(

            request.form["Hourly_Rate"]

        )



        Years_at_Company = float(

            request.form["Years_at_Company"]

        )



        Years_in_Current_Role = float(

            request.form["Years_in_Current_Role"]

        )



        Years_Since_Last_Promotion = float(

            request.form["Years_Since_Last_Promotion"]

        )



        Work_Life_Balance = int(

            request.form["Work_Life_Balance"]

        )



        Job_Satisfaction = int(

            request.form["Job_Satisfaction"]

        )



        Performance_Rating = int(

            request.form["Performance_Rating"]

        )



        Training_Hours_Last_Year = float(

            request.form["Training_Hours_Last_Year"]

        )



        Overtime = request.form["Overtime"]



        Project_Count = int(

            request.form["Project_Count"]

        )



        Average_Hours_Worked_Per_Week = float(

            request.form[

                "Average_Hours_Worked_Per_Week"

            ]

        )



        Absenteeism = float(

            request.form["Absenteeism"]

        )



        Work_Environment_Satisfaction = int(

            request.form[

                "Work_Environment_Satisfaction"

            ]

        )



        Relationship_with_Manager = int(

            request.form[

                "Relationship_with_Manager"

            ]

        )



        Job_Involvement = int(

            request.form["Job_Involvement"]

        )



        Distance_From_Home = float(

            request.form["Distance_From_Home"]

        )



        Number_of_Companies_Worked = int(

            request.form[

                "Number_of_Companies_Worked"

            ]

        )





        # ====================================================

        # CREATE DATAFRAME

        # ====================================================



        input_data = pd.DataFrame({



            "Age": [Age],



            "Gender": [Gender],



            "Marital_Status": [

                Marital_Status

            ],



            "Department": [

                Department

            ],



            "Job_Role": [

                Job_Role

            ],



            "Job_Level": [

                Job_Level

            ],



            "Monthly_Income": [

                Monthly_Income

            ],



            "Hourly_Rate": [

                Hourly_Rate

            ],



            "Years_at_Company": [

                Years_at_Company

            ],



            "Years_in_Current_Role": [

                Years_in_Current_Role

            ],



            "Years_Since_Last_Promotion": [

                Years_Since_Last_Promotion

            ],



            "Work_Life_Balance": [

                Work_Life_Balance

            ],



            "Job_Satisfaction": [

                Job_Satisfaction

            ],



            "Performance_Rating": [

                Performance_Rating

            ],



            "Training_Hours_Last_Year": [

                Training_Hours_Last_Year

            ],



            "Overtime": [

                Overtime

            ],



            "Project_Count": [

                Project_Count

            ],



            "Average_Hours_Worked_Per_Week": [

                Average_Hours_Worked_Per_Week

            ],



            "Absenteeism": [

                Absenteeism

            ],



            "Work_Environment_Satisfaction": [

                Work_Environment_Satisfaction

            ],



            "Relationship_with_Manager": [

                Relationship_with_Manager

            ],



            "Job_Involvement": [

                Job_Involvement

            ],



            "Distance_From_Home": [

                Distance_From_Home

            ],



            "Number_of_Companies_Worked": [

                Number_of_Companies_Worked

            ]

        })





        # ====================================================

        # ENCODE BINARY COLUMNS

        # ====================================================



        input_data["Gender"] = (

            input_data["Gender"]

            .map({

                "Female": 0,

                "Male": 1

            })

        )



        input_data["Overtime"] = (

            input_data["Overtime"]

            .map({

                "No": 0,

                "Yes": 1

            })

        )





        # ====================================================

        # FEATURE ENGINEERING

        # SAME AS TRAINING

        # ====================================================



        # Income per Job Level



        input_data[

            "IncomePerJobLevel"

        ] = (

            input_data["Monthly_Income"]

            /

            input_data["Job_Level"]

            .replace(0, np.nan)

        )





        # Experience Ratio



        input_data[

            "ExperienceRatio"

        ] = (

            input_data["Years_in_Current_Role"]

            /

            input_data["Years_at_Company"]

            .replace(0, np.nan)

        )





        # Promotion Delay Flag



        input_data[

            "Promotion_Delay_Flag"

        ] = (

            input_data[

                "Years_Since_Last_Promotion"

            ] >= 4

        ).astype(int)





        # ====================================================

        # ONE-HOT ENCODING

        # SAME AS TRAINING

        # ====================================================



        input_data = pd.get_dummies(

            input_data,

            columns=[

                "Marital_Status",

                "Department",

                "Job_Role"

            ],

            drop_first=True

        )





        # ====================================================

        # MATCH TRAINING FEATURES

        # ====================================================



        input_data = input_data.reindex(

            columns=feature_names,

            fill_value=0

        )





        # ====================================================

        # HANDLE MISSING VALUES

        # ====================================================



        input_data = input_data.replace(

            [np.inf, -np.inf],

            np.nan

        )



        input_data = input_data.fillna(0)





        # ====================================================

        # SCALE DATA

        # ====================================================



        input_scaled = scaler.transform(

            input_data

        )





        # ====================================================

        # PREDICTION

        # ====================================================



        prediction = model.predict(

            input_scaled

        )



        probability = model.predict_proba(

            input_scaled

        )[0][1]





        # ====================================================

        # ATTRITION RESULT

        # ====================================================



        risk_percentage = round(

            probability * 100,

            2

        )





        if prediction[0] == 1:



            prediction_text = (

                "High Attrition Risk"

            )



        else:



            prediction_text = (

                "Low Attrition Risk"

            )





        # ====================================================

        # RISK LEVEL

        # ====================================================



        if probability >= 0.70:



            risk_level = "High Risk"



        elif probability >= 0.40:



            risk_level = "Medium Risk"



        else:



            risk_level = "Low Risk"





        # ====================================================

        # SHAP EXPLANATION

        # ====================================================



        shap_values = explainer.shap_values(

            input_scaled

        )





        # Handle different SHAP output formats



        if isinstance(shap_values, list):



            employee_shap = (

                shap_values[1][0]

            )



        else:



            shap_array = np.asarray(

                shap_values

            )



            if shap_array.ndim == 3:



                employee_shap = (

                    shap_array[0, :, 1]

                )



            elif shap_array.ndim == 2:



                employee_shap = (

                    shap_array[0]

                )



            else:



                employee_shap = (

                    shap_array.flatten()

                )





        # ====================================================

        # TOP 5 SHAP FEATURES

        # ====================================================



        shap_df = pd.DataFrame({



            "Feature": feature_names,



            "SHAP": np.asarray(

                employee_shap

            ).flatten()



        })





        shap_df["Absolute_SHAP"] = (

            shap_df["SHAP"].abs()

        )





        top_features = (

            shap_df

            .sort_values(

                by="Absolute_SHAP",

                ascending=False

            )

            .head(5)

        )





        # ====================================================

        # HR RECOMMENDATIONS

        # ====================================================



        recommendations = []





        for feature in top_features[

            "Feature"

        ]:



            if feature in recommendation_mapping:



                recommendations.append(

                    recommendation_mapping[

                        feature

                    ]

                )





        # ====================================================

        # SEND RESULT TO HTML

        # ====================================================



        return render_template(



            "index.html",



            prediction_text=prediction_text,



            risk_level=risk_level,



            risk_percentage=risk_percentage,



            recommendations=recommendations



        )





    # ========================================================

    # ERROR HANDLING

    # ========================================================



    except Exception as e:



        print("====================================")

        print("PREDICTION ERROR")

        print("Error type:", type(e).__name__)

        print("Error message:", str(e))

        print("Received form fields:")

        print(list(request.form.keys()))

        print("====================================")



        return render_template(

            "index.html",

            prediction_text=f"Error: {type(e).__name__}: {str(e)}"

        )





# ============================================================

# RUN FLASK APP

# ============================================================



if __name__ == "__main__":



    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )
