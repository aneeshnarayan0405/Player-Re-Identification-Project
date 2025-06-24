from sklearn.metrics.pairwise import cosine_similarity

class ReIDTracker:
    def __init__(self, similarity_threshold=0.75):
        self.players = {}
        self.next_id = 0
        self.similarity_threshold = similarity_threshold

    def update(self, features, boxes):
        assigned_ids = []
        for i, feat in enumerate(features):
            best_id = None
            best_sim = self.similarity_threshold
            for pid, data in self.players.items():
                sim = cosine_similarity([feat], [data['feature']])[0][0]
                if sim > best_sim:
                    best_id = pid
                    best_sim = sim
            if best_id is not None:
                assigned_ids.append(best_id)
                self.players[best_id]['feature'] = feat
            else:
                assigned_ids.append(self.next_id)
                self.players[self.next_id] = {'feature': feat}
                self.next_id += 1
        return assigned_ids
