
MMLU_SUBSETS = [
    "college_medicine", "college_biology", "college_chemistry", "college_physics",
    "clinical_knowledge", "professional_medicine", "anatomy", "abstract_algebra",
    "astronomy", "business_ethics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "high_school_biology", "high_school_chemistry", "high_school_physics",
    "high_school_mathematics", "high_school_computer_science", "high_school_government",
    "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "virology", "world_religions",
]

MMLU_SUBSET_GROUPS = [
    {"id": "biology_medicine", "label": "生物/医学", "subsets": [
        "college_medicine", "college_biology", "clinical_knowledge", "professional_medicine",
        "anatomy", "medical_genetics", "virology", "high_school_biology", "human_aging", "human_sexuality",
    ]},
    {"id": "stem", "label": "STEM（数理/工程/计算机）", "subsets": [
        "college_chemistry", "college_physics", "abstract_algebra", "astronomy", "computer_security",
        "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_chemistry",
        "high_school_physics", "high_school_mathematics", "high_school_computer_science", "machine_learning",
    ]},
    {"id": "humanities_social", "label": "人文/法律/社科", "subsets": [
        "high_school_government", "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history",
        "international_law", "jurisprudence", "philosophy", "prehistory", "sociology", "us_foreign_policy", "world_religions",
    ]},
    {"id": "business_economics", "label": "经济/商科", "subsets": [
        "business_ethics", "econometrics", "management", "marketing", "professional_accounting",
    ]},
    {"id": "other", "label": "其他", "subsets": [
        "logical_fallacies", "miscellaneous", "moral_disputes", "moral_scenarios",
        "professional_law", "professional_psychology", "public_relations", "security_studies",
    ]},
]

CMMMU_SUBSETS = [
    "art_and_design", "business", "health_and_medicine",
    "humanities_and_social_sciences", "science", "technology_and_engineering",
]

CMMMU_SUBSET_GROUPS = [
    {"id": "health_medicine", "label": "健康与医学", "subsets": ["health_and_medicine"]},
    {"id": "stem", "label": "STEM", "subsets": ["science", "technology_and_engineering"]},
    {"id": "humanities_art", "label": "人文与艺术", "subsets": ["art_and_design", "humanities_and_social_sciences"]},
    {"id": "business", "label": "商学", "subsets": ["business"]},
]
