"""
Career Knowledge Base
----------------------
Each entry represents a career profile. The 'description' field is what gets
converted into a vector (embedding) using a pretrained SBERT model, so it
should be written in natural language describing the work, skills, and
personality traits that suit that career.

'key_subjects' + 'min_scores' are used for the academic-fit rule layer
(non-ML, transparent scoring logic) that boosts/penalizes careers based on
the user's entered academic scores.

PROJECT STAGE (Week 6): This is a partial draft knowledge base — 10 career
profiles across Science, Commerce, and Arts streams — used to validate the
SBERT-based matching approach. Expansion to 35+ profiles is planned for
Month 2 (see project Work Plan).
"""

CAREERS = [
    {
        "name": "Data Scientist",
        "stream": "Science",
        "description": "Works with large datasets to find patterns, builds machine learning models, uses statistics and programming in Python to solve business problems. Requires strong logical thinking, curiosity, and comfort with math and coding.",
        "key_subjects": ["Math", "Computer Science"],
        "min_scores": {"Math": 65, "Computer Science": 60},
    },
    {
        "name": "Software Engineer",
        "stream": "Science",
        "description": "Designs, builds, and maintains software applications and systems. Involves problem solving, writing clean code, debugging, and working with teams to ship products. Suits people who enjoy logic puzzles and building things.",
        "key_subjects": ["Math", "Computer Science"],
        "min_scores": {"Math": 55, "Computer Science": 60},
    },
    {
        "name": "Mechanical Engineer",
        "stream": "Science",
        "description": "Designs and analyzes mechanical systems, machines, and tools using physics and engineering principles. Involves CAD software, thermodynamics, and hands-on problem solving for manufacturing and design.",
        "key_subjects": ["Math", "Physics"],
        "min_scores": {"Math": 60, "Physics": 60},
    },
    {
        "name": "Doctor (Medicine)",
        "stream": "Science",
        "description": "Diagnoses and treats illness, cares for patients, and requires deep knowledge of biology and human anatomy. Suits people who are empathetic, detail-oriented, and interested in health sciences.",
        "key_subjects": ["Biology", "Chemistry"],
        "min_scores": {"Biology": 70, "Chemistry": 60},
    },
    {
        "name": "AI/ML Engineer",
        "stream": "Science",
        "description": "Builds and deploys machine learning and artificial intelligence systems, including neural networks and NLP models, to solve real-world problems. Requires strong math, programming, and analytical thinking.",
        "key_subjects": ["Math", "Computer Science"],
        "min_scores": {"Math": 65, "Computer Science": 65},
    },
    {
        "name": "Chartered Accountant",
        "stream": "Commerce",
        "description": "Manages financial records, audits accounts, prepares tax filings, and advises businesses on financial strategy. Requires strong numerical ability, attention to detail, and integrity.",
        "key_subjects": ["Math", "Accountancy"],
        "min_scores": {"Math": 55},
    },
    {
        "name": "Marketing Manager",
        "stream": "Commerce",
        "description": "Plans and executes campaigns to promote products or services, studies consumer behavior, and manages brand strategy. Suits creative, people-oriented individuals who enjoy communication and trends.",
        "key_subjects": ["Business Studies"],
        "min_scores": {},
    },
    {
        "name": "Lawyer",
        "stream": "Arts",
        "description": "Represents clients in legal matters, interprets laws, and argues cases in court or advises on legal compliance. Requires strong reading comprehension, argumentation skills, and attention to detail.",
        "key_subjects": ["English", "Political Science"],
        "min_scores": {"English": 60},
    },
    {
        "name": "Graphic Designer",
        "stream": "Arts",
        "description": "Creates visual content for branding, advertising, and digital media using design software. Suits creative individuals with a strong sense of aesthetics and visual storytelling.",
        "key_subjects": [],
        "min_scores": {},
    },
    {
        "name": "Teacher / Educator",
        "stream": "Arts",
        "description": "Educates students in a specific subject, plans lessons, and mentors young learners. Suits patient, communicative people who enjoy helping others learn and grow.",
        "key_subjects": [],
        "min_scores": {},
    },
]