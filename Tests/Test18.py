names = ["SegNet", "U-Net++", "TransUNet", "TransAttUNet", "DeepLabV3", "UPPAQCES", "DP-U-Net++"]
params = [29444162, 33698479, 107533986, 25965899, 58035776, 17456186, 29040721]
sizes = [752, 258, 808, 185, 443, 131, 216]

ratios = [round(a / sizes[i], 2) for i, a in enumerate(params)]

print(ratios)