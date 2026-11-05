
from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app = Flask(__name__)


def validate_student_inputs(form):

    # Check that all required fields are provided
    required_fields = ["studytime", "absences", "G1", "G2"]

    for field in required_fields:
        if not form.get(field, "").strip():
            raise ValueError(f"Please provide a value for {field}.")

    try:
        studytime = int(form.get("studytime"))
        absences = int(form.get("absences"))
        G1 = int(form.get("G1"))
        G2 = int(form.get("G2"))

    except ValueError:
        raise ValueError("All inputs must be valid whole numbers.")

    # Validate study time
    if studytime not in [1, 2, 3, 4]:
        raise ValueError(
            "Study time must be selected from the available options."
        )

    # Validate absences
    if absences < 0:
        raise ValueError(
            "Number of absences cannot be negative."
        )

    # Validate first period grade
    if G1 < 0 or G1 > 20:
        raise ValueError(
            "First period grade (G1) must be between 0 and 20."
        )

    # Validate second period grade
    if G2 < 0 or G2 > 20:
        raise ValueError(
            "Second period grade (G2) must be between 0 and 20."
        )

    return studytime, absences, G1, G2


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:

            # Validate submitted student information
            studytime, absences, G1, G2 = validate_student_inputs(
                request.form
            )

            # Create student data
            student_data = CustomData(
                studytime=studytime,
                absences=absences,
                G1=G1,
                G2=G2
            )

            # Convert inputs into a DataFrame
            student_df = student_data.to_dataframe()

            # Generate prediction
            predict_pipeline = PredictPipeline()
            prediction = round(
                predict_pipeline.predict(student_df), 2
            )

        except ValueError as e:
            error = str(e)

        except Exception:
            error = (
                "Something went wrong while generating the prediction. "
                "Please try again."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
