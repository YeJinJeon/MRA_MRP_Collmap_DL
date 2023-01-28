origin = "./dataset/dasan_dce_data.csv"
new = "./dataset/yes_dasan_dce_data.csv"

origin_dataset = []
with open(origin, "r") as f1:
    for f in f1:
        origin_dataset.append(f.split(",")[0])

new_dataset = []
with open(new, "r") as f1:
    for f in f1:
        new_dataset.append(f.split(",")[0])

no_exist = set(new_dataset).difference(set(origin_dataset))
cmc = 0 
ku = 0
for i in no_exist:
    if "CMC_DATA" in i:
        cmc+=1
    else:
        ku +=1
print(cmc)
print(ku)

output_file = "./dataset/dasan2_dce_data.csv"

with open(output_file, 'w') as f:
    for a in no_exist:
        f.write(f"{a},2\n")



