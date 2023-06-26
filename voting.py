import csv

def vote(mode):
    votes = {}
    with open("./results.csv", 'r') as file:
        csvreader = csv.reader(file)
        next(csvreader)

        if mode == "majority":
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
            print(value_rows)

            
            with open('majority_voting.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header_row)
                writer.writerows(value_rows)
                