
import os
import random

# DIrectory for text files
text_dir = "news_paper_text_files"


# Sample topics and sentences
TOPICS = {
    "sports": [
        "The football match ended with a dramatic penalty shootout.",
        "Basketball players train daily to improve their skills.",
        "Tennis tournaments attract thousands of fans worldwide.",
        "Olympic athletes dedicate years to perfecting their craft.",
        "Soccer fans celebrated wildly after their team's victory.",
        "Swimming is one of the most physically demanding sports.",
        "The championship game was postponed due to heavy rain.",
        "Cricket is immensely popular in countries like India and Australia.",
        "Gymnasts amazed the crowd with their flawless routines.",
        "Marathon runners push their limits to cross the finish line.",
        "Esports are gaining recognition as competitive sports.",
        "The underdog team surprised everyone by winning the title.",
        "Winter sports like skiing require exceptional balance.",
        "Boxing matches often end with knockout punches.",
        "Sports scholarships help students pursue higher education.",
    ],
    "politics": [
        "The government announced new policies for economic growth.",
        "Elections will be held next month across the country.",
        "The president addressed the nation regarding climate change.",
        "Diplomatic talks between the two nations resumed this week.",
        "The new law aims to reduce carbon emissions by 2030.",
        "Protests erupted in the capital demanding political reform.",
        "The opposition party criticized the budget proposal.",
        "International leaders gathered for the climate summit.",
        "Voter turnout reached a record high this election.",
        "The prime minister pledged to improve healthcare access.",
        "A trade agreement was signed to boost bilateral relations.",
        "Corruption scandals dominated the headlines this year.",
        "The Supreme Court ruled on the controversial case.",
        "Tax reforms sparked debates among lawmakers.",
        "Human rights activists condemned the new policy.",
    ],
    "technology": [
        "AI is revolutionizing industries with automation.",
        "Quantum computing is the next big leap in technology.",
        "Cybersecurity threats are increasing in the digital age.",
        "Self-driving cars are expected to reduce traffic accidents.",
        "Blockchain technology ensures secure financial transactions.",
        "5G networks promise faster internet speeds globally.",
        "Tech companies are investing heavily in renewable energy.",
        "Augmented reality is transforming the gaming industry.",
        "Robotics is being used to perform complex surgeries.",
        "Smart home devices are becoming more affordable.",
        "SpaceX launched another batch of satellites into orbit.",
        "Biometric authentication is replacing traditional passwords.",
        "The metaverse is redefining virtual social interactions.",
        "Nanotechnology could revolutionize medical treatments.",
        "Open-source software promotes collaborative development.",
    ],
    "health": [
        "A balanced diet is essential for a healthy lifestyle.",
        "Doctors recommend at least 30 minutes of exercise daily.",
        "Mental health awareness is growing in modern society.",
        "Vaccination campaigns have reduced disease outbreaks.",
        "Yoga and meditation help reduce stress levels.",
        "Researchers discovered a potential cure for a rare disease.",
        "Sleep deprivation can lead to serious health issues.",
        "Telemedicine allows patients to consult doctors remotely.",
        "Antibiotic resistance is a major global health concern.",
        "Organic food consumption is linked to better health.",
        "The pandemic highlighted the importance of public health.",
        "Stem cell therapy offers hope for chronic conditions.",
        "Wearable fitness trackers monitor heart rate and steps.",
        "Sugar consumption is a leading cause of obesity.",
        "Regular health check-ups can prevent severe illnesses.",
    ],
    "entertainment": [
        "Hollywood movies attract audiences worldwide.",
        "The latest music album topped the global charts.",
        "Streaming services are changing how people watch TV.",
        "Award shows celebrate excellence in film and music.",
        "Stand-up comedy specials are trending on platforms.",
        "Broadway shows are returning after a long hiatus.",
        "Celebrity gossip magazines sell millions of copies.",
        "Virtual concerts became popular during the pandemic.",
        "Anime has gained a massive international following.",
        "Podcasts cover topics from true crime to comedy.",
        "Film festivals showcase independent cinema.",
        "Social media influencers dominate digital entertainment.",
        "Classic novels are being adapted into TV series.",
        "Video game tournaments draw huge online audiences.",
        "Reality TV shows remain a guilty pleasure for many.",
    ],
}


def create_directory(directory: str):
    os.makedirs(directory, exist_ok= True)
def generate_text_content(topics: dict, num_sentences: int =10) -> str:
    category = random.choice(list(topics.keys()))
    a = "\n".join(random.choices(topics[category], k= num_sentences))
    return a
def generate_text_files(directory:str, topics:dict, num_files: int =1000, num_sentences: int=10):
    create_directory(directory)

    for i in range(1, num_files +1):
        content = generate_text_content(topics, num_sentences)
        file_path = os.path.join(directory, f"text_file_{i}.txt")
        with open(file_path,"w", encoding="utf-8") as f:
            f.write(content)
    print(f"Generated {num_files} text files in 'directory' directory.")

# Run the file generation
if __name__ == "__main__":
    generate_text_files(text_dir, TOPICS, num_files=1000, num_sentences=10)


