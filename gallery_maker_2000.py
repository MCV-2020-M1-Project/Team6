import pickle as pkl
import cv2 


harmony_paintings = {}

with open('pkl_data/analogos.pkl', 'rb') as f:
    d = pkl.load(f)
    harmony_paintings['analog'] = d

# with open('pkl_data/colorless.pkl', 'rb') as f:
#     d = pkl.load(f)
#     harmony_paintings['colorless'] = d

with open('pkl_data/monochroma.pkl', 'rb') as f:
    d = pkl.load(f)
    harmony_paintings['monochroma'] = d

with open('pkl_data/multicolor.pkl', 'rb') as f:
    d = pkl.load(f)
    harmony_paintings['multicolor'] = d

with open('pkl_data/opposite.pkl', 'rb') as f:
    d = pkl.load(f)
    harmony_paintings['opposite'] = d

# for k, v in harmony_paintings.items():
#     print(k, v)

# Rooms
rooms = {
    'rainbow_warm_room': [],
    'rainbow_cold_room': [],
    'complementary_room': [],
    'analog_room': [],
    'colorfull_room': [],
    'gray_room':[]
}

warm_colors = {'orange', 'red', 'yellow'}
cold_colors = {'turquoise', 'green', 'blue'}
# Rainbow warm and cold
for k, v in harmony_paintings['monochroma'].items():
    if v in warm_colors:
        rooms['rainbow_warm_room'].append(k)
    elif v in cold_colors:
        rooms['rainbow_cold_room'].append(k)
    else:
        print(f'Unknown monochrome color {v}. Not adding to any room')

# Complementary
for k, v in harmony_paintings['opposite'].items():
    # print(k, v)
    rooms['complementary_room'].append(k)

# Analog
for k, v in harmony_paintings['analog'].items():
    rooms['analog_room'].append(k)

# Multicolor
for k, v in harmony_paintings['multicolor'].items():
    rooms['colorfull_room'].append(k)

print('Number of paintings per room:')
for k, v in rooms.items():
    print('\t' + k + ': ' + str(len(v)) + ' paintings')

# Display
show_max = 10
for room_name, paintings in rooms.items():
    for p in paintings[:show_max]:
        im = cv2.imread('../datasets/BBDD/' + p)
        cv2.imshow(room_name, cv2.resize(im, (500, 500*im.shape[0]//im.shape[1])))
        cv2.waitKey(0)

