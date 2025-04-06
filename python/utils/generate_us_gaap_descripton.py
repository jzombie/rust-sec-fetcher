import re


def generate_us_gaap_description(us_gaap_tag_name):
    return re.sub(
        r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", us_gaap_tag_name
    ).lower()
