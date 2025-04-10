import re


def generate_us_gaap_description(us_gaap_concept_name):
    return re.sub(
        r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", us_gaap_concept_name
    ).lower()
