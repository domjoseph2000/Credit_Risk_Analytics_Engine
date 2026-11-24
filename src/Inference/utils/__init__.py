"""
Supporting code for inference: settings, model loading, scoring maths,
explanations, dashboard numbers and the Airflow batch tasks.

Nothing here is run directly. The entry points one folder up
(score_model_input.py, realtime_api.py, batch_dashboard_api.py,
verify_feature_parity.py) and the Airflow batch DAG import from it, e.g.
`from utils import config`.
"""
