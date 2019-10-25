def fileInterpreter(file):
    line = file.readline()

    exec('global results; results = '+line)

    return results

with open('output/predicted.txt', 'r') as predFile, open('output/real.txt', 'r') as realFile:
    predList = fileInterpreter(predFile)
    realList = fileInterpreter(realFile)

    resultsDict = {}
    for real, pred in zip(realList, predList):
        for realTag, predTag in zip(real, pred):
            try:
                if realTag == predTag:
                    resultsDict[realTag]['correct'] += 1
                resultsDict[realTag]['total'] += 1
                resultsDict[realTag]['accuracy'] = resultsDict[realTag]['correct'] / resultsDict[realTag]['total']

            except KeyError:
                resultsDict[realTag] = {}
                resultsDict[realTag]['correct'] = 1 if realTag == predTag else 0
                resultsDict[realTag]['total'] = 1
                resultsDict[realTag]['accuracy'] = resultsDict[realTag]['correct'] / resultsDict[realTag]['total']

    outFile = open("resultsDict.txt","w+")
    outFile.write("%s", resultsDict)
    outFile.close()
