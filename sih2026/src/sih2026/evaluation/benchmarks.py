"""Industry target role benchmarks dataset and lookup module."""

INDUSTRY_BENCHMARKS = {
    "Full-Stack Software Developer": {
        "required_skills": [
            "JavaScript / TypeScript", "React / Next.js", "Node.js / Express",
            "REST APIs", "SQL / Databases", "Git / GitHub", "HTML / CSS",
            "Docker / Containerization", "CI/CD Pipelines", "Testing / Jest / PyTest"
        ],
        "core_concepts": [
            "Web Architecture", "Database Modeling", "State Management",
            "Authentication & Security", "Asynchronous Programming"
        ],
        "emerging_trends": [
            "Serverless Architecture", "GraphQL", "Tailwind CSS", "Microservices"
        ]
    },
    "Data Scientist": {
        "required_skills": [
            "Python", "Pandas & NumPy", "Scikit-Learn", "SQL",
            "Machine Learning", "Data Visualization (Matplotlib/Seaborn)",
            "Deep Learning (PyTorch/TensorFlow)", "Feature Engineering",
            "Statistics & Probability", "MLOps / Model Deployment"
        ],
        "core_concepts": [
            "Supervised vs Unsupervised Learning", "Model Evaluation Metrics",
            "Cross-Validation", "Data Preprocessing", "Exploratory Data Analysis"
        ],
        "emerging_trends": [
            "Large Language Models (LLMs)", "Retrieval-Augmented Generation (RAG)",
            "Vector Databases", "Prompt Engineering"
        ]
    },
    "AI Engineer": {
        "required_skills": [
            "Python", "PyTorch / TensorFlow", "LLMs & Transformers",
            "Vector Databases (Chroma / Qdrant)", "LangChain / LlamaIndex",
            "RAG Pipelines", "API Integration", "Model Fine-Tuning",
            "Docker / Deployment", "Git"
        ],
        "core_concepts": [
            "Neural Networks", "Attention Mechanism", "Embeddings & Vector Search",
            "Prompt Engineering", "Evaluation Metrics for Generative AI"
        ],
        "emerging_trends": [
            "AI Agents & Tool Use", "Small Language Models (SLMs)", "Multimodal AI"
        ]
    },
    "DevOps Engineer": {
        "required_skills": [
            "Linux Systems Administration", "Docker", "Kubernetes",
            "CI/CD (GitHub Actions / Jenkins)", "Infrastructure as Code (Terraform / Ansible)",
            "Cloud Platforms (AWS / GCP / Azure)", "Python / Bash Scripting",
            "Monitoring & Logging (Prometheus / Grafana)", "Networking & Security"
        ],
        "core_concepts": [
            "Container Orchestration", "Continuous Integration", "GitOps",
            "Site Reliability Engineering (SRE)", "Infrastructure Automation"
        ],
        "emerging_trends": [
            "Platform Engineering", "Cloud-Native Security", "eBPF Monitoring"
        ]
    },
    "Cybersecurity Analyst": {
        "required_skills": [
            "Network Security", "Ethical Hacking & Penetration Testing",
            "SIEM Tools (Splunk / Elastic)", "Cryptography", "Python / Bash Scripting",
            "Incident Response", "Vulnerability Management", "Linux / Windows Security"
        ],
        "core_concepts": [
            "CIA Triad", "Zero Trust Architecture", "Threat Modeling",
            "Firewalls & IDS/IPS", "Security Auditing"
        ],
        "emerging_trends": [
            "AI-driven Threat Detection", "Cloud Security Posture Management (CSPM)"
        ]
    }
}


def get_industry_benchmarks(role_title: str = "Full-Stack Software Developer") -> dict:
    """Returns required skills, core concepts, and emerging trends for a tech role."""
    clean_role = role_title.strip()
    for role_name, profile in INDUSTRY_BENCHMARKS.items():
        if role_name.lower() == clean_role.lower():
            return {"role": role_name, **profile}

    # Partial match fallback
    for role_name, profile in INDUSTRY_BENCHMARKS.items():
        if clean_role.lower() in role_name.lower() or role_name.lower() in clean_role.lower():
            return {"role": role_name, **profile}

    # Default fallback
    return {
        "role": clean_role,
        "required_skills": ["Core Programming", "Data Structures", "Algorithms", "Databases", "Version Control"],
        "core_concepts": ["Software Design", "Problem Solving", "System Architecture"],
        "emerging_trends": ["AI Tools", "Cloud Computing"]
    }
