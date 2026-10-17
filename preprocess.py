import pandas as pd

df = pd.read_csv("RECONSTRUCTED trajectories-400-0415_NO MOTORCYCLES.csv")
df["Local_Y"] *= 0.3048
df["Mean_Speed"] *= 0.3048
df["Vehicle_Length"] *= 0.3048

processedRows = []

df.set_index(["Vehicle_ID", "Frame_ID"], inplace = True) #set index so vehicle at specific frame can be accessed with df.loc

for (followerID, frameID), row in df.iterrows(): #for each condiiton extract dynamics

    leaderID = row["Leader_ID"]

    if leaderID == 0:
        continue

    try:
        leaderRow = df.loc[(leaderID, frameID)] #find leader vehicle data at same timestamp

        followSpeed = row["Mean_Speed"]
        followPos = row["Local_Y"]

        leadSpeed = leaderRow["Mean_Speed"]
        leaderPos = leaderRow["Local_Y"]
        leaderLength = leaderRow["Vehicle_Length"]

        spacing = leaderPos - leaderLength - followPos

        processedRows.append({
            "Frame_ID": frameID,
            "Follower_ID": followerID,
            "Follower_Speed": followSpeed,
            "Follower_Position": followPos,
            "Leader_ID": leaderID,
            "Leader_Speed": leadSpeed,
            "Leader_Position": leaderPos,
            "Spacing": spacing
        })

    except KeyError:
        continue

processedDF = pd.DataFrame(processedRows)
processedDF.to_csv("processed_ngsim.csv", index=False)