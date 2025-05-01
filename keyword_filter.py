"""
Keyword filter for humanoid robotics news.

This module contains keyword lists and filtering logic for identifying relevant
humanoid robotics articles.
"""

from typing import List, Dict, Tuple
import re
from collections import defaultdict

# Core keywords that must always pass
HUMANOID_CORE: List[str] = [
    "humanoid robot",
    "android robot",
    "biped robot",
    "bipedal robot",
    "bipedal walker",
    "anthropomorphic robot",
    "factory avatar",
    "robot avatar",
    "whole-body control",
    "whole body impedance",
    "balance recovery",
    "dual-arm manipulation",
    "exoskeleton full-body",
    "exoskeleton lower-limb",
    "Optimus",
    "Tesla Optimus",
    "Optimus Gen 2",
    "Figure AI",
    "Figure 01",
    "Agility Robotics",
    "Digit robot",
    "Apptronik",
    "Apollo robot",
    "Sanctuary AI",
    "Phoenix robot",
    "OpenAI × 1X",
    "UBTECH Walker X",
    "Xiaomi CyberOne",
    "Musashi AI robots",
    "Fourier X20",
    "PND Robotics Adam",
    "Diligent Robots biped",
    "Toyota T-HR3",
    "Honda Asimo",
    "Atlas",
    "RE2 Sapien biped",
    "ICRA Humanoid",
    "IEEE RAS TC Humanoids"
]

# Enabling technologies that add value if paired with core keywords
ENABLING_TECH: Dict[str, List[str]] = {
    "Actuation & Locomotion": [
        "series-elastic actuator",
        "high-torque actuator",
        "harmonic drive",
        "cycloidal gearbox",
        "quasi-direct drive",
        "torque sensor",
        "force-controlled joint",
        "ankle push-off",
        "zero-moment point",
        "model-predictive gait",
        "variable stiffness",
        "toe-off mechanism"
    ],
    "Power, Batteries, Energy": [
        "4680 cell",
        "4680-LFP",
        "cobalt-free battery",
        "silicon-anode",
        "solid-state pack",
        "SiC inverter",
        "bidirectional DC/DC",
        "fast-swap battery module",
        "regenerative braking",
        "energy-efficient locomotion",
        "12 kg-h/J"
    ],
    "Compute & AI Hardware": [
        "Jetson Orin",
        "Jetson Thor",
        "Xavier NX",
        "AMD Versal AI Edge",
        "Intel Core Ultra",
        "Meteor Lake",
        "Qualcomm RB-series",
        "ARM Neoverse",
        "FPGA gait controller",
        "realtime ROS 2",
        "micro-ROS",
        "EtherCAT motion bus",
        "TSN robot"
    ],
    "Sensing & Perception": [
        "event-based camera",
        "tactile skin",
        "e-skin",
        "mmWave radar",
        "force-torque wrist",
        "3-D LIDAR",
        "depth-AI module",
        "visual-locomotion fusion",
        "SLAM biped"
    ],
    "Materials & Manufacturing": [
        "lightweight carbon composite",
        "3-D printed titanium",
        "injection-moulded structural battery",
        "recyclable robot chassis",
        "modular limb architecture"
    ]
}

# India-specific context keywords
INDIA_CONTEXT: Dict[str, List[str]] = {
    "Government & Policy": [
        "DRDO",
        "ADRDE",
        "ARDE",
        "ISRO Humanoid",
        "MeitY",
        "DPIIT",
        "PLI scheme",
        "Make in India robotics",
        "Digital India Robotics Summit",
        "SAMARTH-Udyog",
        "iDEX",
        "Technology Development Board",
        "Atal Tinkering Lab"
    ],
    "Academic & Research": [
        "IIT Madras RRC",
        "IIT Madras TTRP",
        "IIT Kanpur Humanoid Lab",
        "IIT Bombay RuBa",
        "IISc Bengaluru Biorobotics",
        "IIIT Hyderabad Robotics Lab",
        "BITS Pilani DRDO Centre",
        "C-DAC robotics"
    ],
    "Industry & Start-ups": [
        "Tata Advanced Systems humanoid",
        "Mahindra Robotics",
        "Adani Defence & Aerospace robots",
        "Sastra Robotics humanoid",
        "Genrobotics Humanoid",
        "Asimov Robotics Kochi",
        "Invento Makie",
        "GreyOrange biped",
        "Omnipresent Robotics human-assist",
        "Botlab Dynamics humanoid"
    ]
}

# Negative keywords that should be filtered out unless paired with core keywords
NEGATIVE_KEYWORDS: List[str] = [
    "surgical robot",
    "warehouse AMR",
    "drone delivery",
    "EV scooter",
    "autonomous car",
    "vacuum cleaner robot",
    "swarm drone",
    "agri-drone",
    "toy robot"
]

class KeywordFilter:
    def __init__(self):
        self.core_patterns = self._create_patterns(HUMANOID_CORE)
        self.enabling_tech_patterns = {
            category: self._create_patterns(keywords)
            for category, keywords in ENABLING_TECH.items()
        }
        self.india_context_patterns = {
            category: self._create_patterns(keywords)
            for category, keywords in INDIA_CONTEXT.items()
        }
        self.negative_patterns = self._create_patterns(NEGATIVE_KEYWORDS)

    def get_fact_sheet(self, text: str) -> Dict:
        """
        Generate a fact sheet for article categorization.
        
        Args:
            text: The text to analyze
            
        Returns:
            A dictionary containing:
            - core_match: Whether core keywords were found
            - enabling_tech: List of matching enabling technologies
            - india_context: List of matching India context keywords
            - negative_match: Whether negative keywords were found
        """
        fact_sheet = {
            'core_match': False,
            'enabling_tech': [],
            'india_context': [],
            'negative_match': False
        }
        
        # Check core keywords
        if any(pattern.search(text) for pattern in self.core_patterns):
            fact_sheet['core_match'] = True
        
        # Check enabling technologies
        for category, patterns in self.enabling_tech_patterns.items():
            matches = [pattern for pattern in patterns if pattern.search(text)]
            if matches:
                fact_sheet['enabling_tech'].extend(matches)
        
        # Check India context
        for category, patterns in self.india_context_patterns.items():
            matches = [pattern for pattern in patterns if pattern.search(text)]
            if matches:
                fact_sheet['india_context'].extend(matches)
        
        # Check negative keywords
        if any(pattern.search(text) for pattern in self.negative_patterns):
            fact_sheet['negative_match'] = True
        
        return fact_sheet
        
    def _create_patterns(self, keywords: List[str]) -> List[re.Pattern]:
        """Create regex patterns from keywords with word boundaries."""
        return [re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE) for keyword in keywords]
    
    def is_relevant(self, text: str) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Check if text is relevant based on keyword filters.
        
        Returns:
            Tuple of (is_relevant, match_details)
            where match_details contains lists of matched keywords in each category
        """
        matches = defaultdict(list)
        is_core_match = False
        
        # Check core keywords
        for pattern in self.core_patterns:
            if pattern.search(text):
                matches['core'].append(pattern.pattern)
                is_core_match = True
        
        # Check enabling technologies if we have a core match
        if is_core_match:
            for category, patterns in self.enabling_tech_patterns.items():
                for pattern in patterns:
                    if pattern.search(text):
                        matches['enabling_tech'].append(pattern.pattern)
        
        # Check India context
        for category, patterns in self.india_context_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    matches['india_context'].append(pattern.pattern)
        
        # Check negative keywords
        has_negative = any(pattern.search(text) for pattern in self.negative_patterns)
        
        # Determine relevance
        is_relevant = (
            is_core_match or  # Always relevant if core match
            (matches['enabling_tech'] and matches['india_context']) or  # Relevant if both tech and India context
            (matches['india_context'] and not has_negative)  # Relevant if India context and no negative keywords
        )
        
        return is_relevant, dict(matches)
