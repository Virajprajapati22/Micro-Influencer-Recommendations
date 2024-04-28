import numpy as np
from scipy.spatial.distance import cosine

def calculate_engagement_score(likes, comments, follower):
    # Calculate the average of likes and comments and finally divide it by number of followers
    return np.average(likes + comments) / follower

def calculate_similarity(visual_representation, textual_representation):
    # Calculate cosine similarity between the two representations
    similarity = 1 - cosine(visual_representation, textual_representation)
    return similarity

def calculate_competence_score(engagement_score, similarity_score, alpha):
    # alpha is to set the trade-off between the engagement and similarity scores
    return (alpha * engagement_score) + ((1-alpha) * similarity_score)