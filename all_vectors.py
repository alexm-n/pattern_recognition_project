import os

doss = 'C:/Users/user/OneDrive/Bureau/AnUniv/m1-p/reconnaissance formes/proj'
approche={}
x=[]

for dossier in os.listdir(doss):
    if dossier in ["F0","E34","F2","GFD","SA"]:
        for fichier in os.listdir(os.path.join(doss,dossier)):
            c,e=fichier.capitalize().split('n')
            _,cls = c.split('S')
            ech, _ = e.split('.')

            if dossier not in approche:
                approche[dossier] = {}
            x=[]
            fichier_path = os.path.join(doss,dossier,fichier)

            if os.path.exists(fichier_path):
                with open(fichier_path,"r",encoding="utf-8") as f:
                    for line in f:
                        x.append(float(line))

            approche[dossier][int(cls),int(ech)] = x

print(approche["F0"][1,1])