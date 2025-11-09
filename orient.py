import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import tqdm

last_pt = [np.clip(int(i), 0, 512) for i in sys.argv[1:3]]

def orient_all(last_pt, spline_dict):
    out_dict = {}
    for i in range(len(spline_dict)):
        data = spline_dict[i]
        data_arr = np.array(data)
        dist_unflipped = np.linalg.norm(data_arr[0] - last_pt)
        dist_flip = np.linalg.norm(data_arr[-1] - last_pt)
        if dist_flip < dist_unflipped:
            data = data[::-1]
        last_pt = data[0]
        out_dict[i] = data[:350]
    return out_dict

def main():

    #reads spline
    with open('spline.json', 'r') as f:
        spline_dict = json.load(f)
    spline_dict = {int(k): v for k, v in spline_dict.items()}

    #orients all
    out_dict = orient_all(last_pt, spline_dict)

    #saves outputs
    with open('oriented.json', 'w') as f:
        json.dump(out_dict, f, indent=4)

    #plots oriented spline
    plt.imshow(np.ones((512, 512)), cmap='gray', vmin=0, vmax=1)
    cmap = plt.get_cmap('viridis')
    for i in tqdm.tqdm(range(len(spline_dict))):
        y, x = np.array(out_dict[i]).T
        color = cmap(i / (len(spline_dict) - 1))
        plt.scatter(x[0], y[0], color=color)
        plt.plot(x, y, color=color)
    plt.tight_layout()
    plt.savefig('oriented.png')

if __name__ == '__main__':
    main()