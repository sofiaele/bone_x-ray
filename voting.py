import csv

def vote(mode):
    votes = {}

    if mode == "majority":
        with open("./results.csv", 'r') as file:
            csvreader = csv.reader(file)
            next(csvreader)
    
            for row in csvreader:
                prediction = row[2]
                truth = row[3]

                path = row[1]
                path = path.split('/')
                study = path[2] + '-' + path[3]
                if study in votes: 
                    if prediction == 1:
                        new_votes = votes[study][0] + 1
                    else:
                        new_votes = votes[study][0] - 1

                    if new_votes > 0:
                        votes[study] = [new_votes, 1, truth]
                    else:
                        votes[study] = [new_votes, 0, truth]
                else:
                    if prediction == 1:
                        votes[study] = [1, 1, truth]
                    else:
                        votes[study] = [-1, 0, truth]
            studies = list(votes.keys())
            values = list(votes.values())

            header_row = ['Study', 'Votes', 'Prediction', 'Truth']
            value_rows = [[studies[i]] + values[i] for i in range(len(studies))]
            # print(value_rows)


        with open('majority_voting.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header_row)
            writer.writerows(value_rows)
    elif mode == "soft":
        with open("./results_probs.csv", 'r') as file:
            csvreader = csv.reader(file)
            next(csvreader)

            for row in csvreader:
                prediction_probs = (row[2]).strip('][').split(', ')
                prediction_probs = [float(prediction_probs[0]), float(prediction_probs[1])]
                
                truth = row[3]

                path = row[1]
                path = path.split('/')
                study = path[2] + '-' + path[3]
                
                if study in votes:
                    votes[study] =[ [votes[study][0][0] + prediction_probs[0], votes[study][0][1] + prediction_probs[1]], None, truth, votes[study][3]+1 ]   
                else:
                    votes[study] = [ prediction_probs, None, truth, 1]
                                        
            for study in votes:

                if votes[study][0][0] >= votes[study][0][1]:
                    votes[study] = [  [votes[study][0][0]/votes[study][3], votes[study][0][1]/votes[study][3] ] , 0, votes[study][2], votes[study][3] ]
                else:
                    votes[study] = [  [votes[study][0][0]/votes[study][3], votes[study][0][1]/votes[study][3] ] , 1, votes[study][2], votes[study][3] ]

        
        with open('soft_voting.csv', 'w', newline='') as file:
            header_row = ['Study', 'Votes_Prob', 'Prediction', 'Truth', '#Images']
                          
            studies = list(votes.keys())
            values = list(votes.values())

            value_rows = [[studies[i]] + values[i] for i in range(len(studies))]

            writer = csv.writer(file)
            writer.writerow(header_row)
            writer.writerows(value_rows)
        with open('soft_voting.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header_row)
            writer.writerows(value_rows)


                
