import numpy as np

def calculate_engagement_score(likes, comments, follower):
    # Calculate the average of likes and comments and finally divide it by number of followers
    return np.average(likes + comments) / follower

def calculate_similarity_score(em, eb):
    # Compute dot product
    dot_product = np.dot(em, eb.T)
    
    # Compute magnitudes
    magnitude_em = np.linalg.norm(em)
    magnitude_eb = np.linalg.norm(eb)
    
    # Compute cosine similarity
    similarity_score = dot_product / (magnitude_em * magnitude_eb)
    
    return similarity_score

def calculate_competence_score(engagement_score, similarity_score, alpha):
    # alpha is to set the trade-off between the engagement and similarity scores
    return (alpha * engagement_score) + ((1-alpha) * similarity_score)