import pygame
import random
import time
import numpy as np
import math
import tflearn
from tqdm import tqdm
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression


class snakeApp:
    # initialise snake parameters
    dply_wdth = 550
    dply_hght = 550
    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 255)
    wndw_clr = (255, 255, 255)

    #snk_hd = [250, 250]
    #snk_pos = [[250, 250], [240, 250], [230, 250]]
    #apple_pos = [random.randrange(1, 50)*10, random.randrange(1, 50)*10]
    score = 0
    apple_img = pygame.image.load('apple.jpg')
    #bttn_dir = 0

    def starting_positions(self):
        self.snk_hd = [100, 100]
        self.snk_pos = [[100, 100], [90, 100], [80, 100]]
        self.apple_pos = [random.randrange(1, 50)*10, random.randrange(1, 50)*10]
        self.score = 3

        return self.snk_hd, self.snk_pos, self.apple_pos, self.score

    # upon collision with apple
    def cllson_wth_apple(self, apple_pos, score):
        self.apple_pos = [random.randrange(1, 20)*10, random.randrange(1, 20)*10]
        self.score += 1
        return self.apple_pos, self.score

    # upon collision with game window edge
    def cllson_wth_edges(self, snk_hd):
        if self.snk_hd[0] >= self.dply_hght or self.snk_hd[0] \
                <= 0 or self.snk_hd[1] >= self.dply_hght or self.snk_hd[1] <= 0:
            return 1
        else:
            return 0

    # upon snake collision with itself
    def cllson_wth_snk(self, snk_pos):
        self.snk_hd = self.snk_pos[0]
        if self.snk_hd in self.snk_pos[2:]:
            return 1
        else:
            return 0

    def blckd_drctns(self, snk_pos):
        self.crrnt_drctn_vctr = np.array(self.snk_pos[0]) - np.array(self.snk_pos[1])
        self.lft_drctn_vctr = np.array([self.crrnt_drctn_vctr[1], -self.crrnt_drctn_vctr[0]])
        self.rght_drctn_vctr = np.array([-self.crrnt_drctn_vctr[1], self.crrnt_drctn_vctr[0]])

        self.frnt_blckd = self.drctn_blckd(self.snk_pos, self.crrnt_drctn_vctr)
        self.lft_blckd = self.drctn_blckd(self.snk_pos, self.lft_drctn_vctr)
        self.rght_blckd = self.drctn_blckd(self.snk_pos, self.rght_drctn_vctr)

        return self.frnt_blckd, self.lft_blckd, self.rght_blckd

    # game over if collision detected
    def drctn_blckd(self, snk_pos, crrnt_drctn_vctr):
        self.next_step = self.snk_pos[0] + self.crrnt_drctn_vctr
        self.snk_hd = self.snk_pos[0]
        if self.cllson_wth_edges(self.snk_hd) == 1 or \
                self.cllson_wth_snk(self.snk_pos) == 1:
            return 1
        else:
            return 0

    # deploy the snake
    def dply_snk(self, snk_pos):
        for position in self.snk_pos:
            pygame.draw.rect(self.dply, self.green,
                             pygame.Rect(position[0],
                                         position[1], 10, 10))

    # deploy the apple
    def dply_apple(self, apple_pos, apple_img):
        self.dply.blit(self.apple_img, (self.apple_pos[0], self.apple_pos[1]))

    # distance of the apple from the snake
    def apple_dstnc_snk(self, apple_pos, snk_pos):
        return np.linalg.norm(np.array(self.apple_pos) - np.array(self.snk_pos[0]))

    # generate the snake, detect score with apple collisions
    def gnrt_snk(self, snk_hd, snk_pos, apple_pos, bttn_dir, score):
        if self.bttn_dir == 1:
            self.snk_hd[0] += 10
        elif self.bttn_dir == 0:
            self.snk_hd[0] -= 10
        elif self.bttn_dir == 2:
            self.snk_hd[1] += 10
        else:
            self.snk_hd[1] -= 10

        if self.snk_hd == self.apple_pos:
            self.apple_pos, self.score = self.cllson_wth_apple(self.apple_pos, self.score)
            self.snk_pos.insert(0, list(self.snk_hd))

        else:
            self.snk_pos.insert(0, list(self.snk_hd))
            self.snk_pos.pop()

        return self.snk_pos, self.apple_pos, self.score

    # generate next direction for snake
    def nxt_direction(self, snk_pos, angle):
        self.direction = 0
        # print(self.angle)
        if self.angle > 0:
            self.direction = 1
        elif self.angle < 0:
            self.direction = -1
        else:
            self.direction = 0

        self.crrnt_drctn_vctr = np.array(self.snk_pos[0]) - np.array(self.snk_pos[1])
        self.lft_drctn_vctr = np.array([self.crrnt_drctn_vctr[1], -self.crrnt_drctn_vctr[0]])
        self.rght_drctn_vctr = np.array([-self.crrnt_drctn_vctr[1], self.crrnt_drctn_vctr[0]])

        self.nxt_drctn = self.crrnt_drctn_vctr
        if self.direction == -1:
            self.nxt_drctn = self.lft_drctn_vctr
        if self.direction == 1:
            self.nxt_drctn = self.rght_drctn_vctr

        self.bttn_dir = self.gnrt_bttn_drctn(self.nxt_drctn)

        return self.direction, self.bttn_dir

    # determine button press for game
    def gnrt_bttn_drctn(self, nxt_drctn):
        self.bttn_dir = 0
        if self.nxt_drctn.tolist() == [10, 0]:
            self.bttn_dir = 1
        elif self.nxt_drctn.tolist() == [-10, 0]:
            self.bttn_dir = 0
        elif self.nxt_drctn.tolist() == [0, 10]:
            self.bttn_dir = 2
        else:
            self.bttn_dir = 3
        print(self.nxt_drctn.tolist())
        return self.bttn_dir

    # generate angle with apple for training
    def apple_ang(self, snk_pos, apple_pos):
        # print(self.snk_pos)
        # print(self.apple_pos)
        self.apple_drctn = np.array(self.apple_pos) - np.array(self.snk_pos[0])
        self.snk_drctn = np.array(self.snk_pos[0]) - np.array(self.snk_pos[1])

        # normalise the snake and apple direction vectors
        self.norm_apple_drctn = np.linalg.norm(self.apple_drctn)
        self.norm_snk_drctn = np.linalg.norm(self.snk_drctn)
        if self.norm_apple_drctn == 0:
            self.norm_apple_drctn = 10
        if self.norm_snk_drctn == 0:
            self.norm_snk_drctn = 10

        self.nrmlsd_apple = self.apple_drctn / self.norm_apple_drctn
        self.nrmlsd_snk = self.snk_drctn / self.norm_snk_drctn
        # print(self.nrmlsd_snk)
        # print(self.nrmlsd_apple)
        self.angle = math.atan2(self.nrmlsd_apple[1] * self.nrmlsd_snk[0] -
                                self.nrmlsd_apple[0] * self.nrmlsd_snk[1],
                                self.nrmlsd_apple[1] * self.nrmlsd_snk[1] +
                                self.nrmlsd_apple[0] * self.nrmlsd_snk[0]) / math.pi
        return self.angle

    # main game loop
    def game_loop(self, snk_pos, snk_hd, apple, apple_pos, bttn_dir, score):
        self.crash = False
        """self.prev_bttn_dir = 1
        self.bttn_dir = 1
        self.crrnt_drctn_vctr = """
        while self.crash is not True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.crash = True
                """if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and self.prev_bttn_dir != 1:
                        self.bttn_dir = 0
                    elif event.key == pygame.K_RIGHT and self.prev_bttn_dir != 0:
                        self.bttn_dir = 1
                    elif event.key == pygame.K_DOWN and self.prev_bttn_dir != 3:
                        self.bttn_dir = 2
                    elif event.key == pygame.K_UP and self.prev_bttn_dir != 2:
                        self.bttn_dir = 3
                    else:
                        self.bttn_dir = self.bttn_dir"""
            self.dply.fill(self.wndw_clr)
            self.dply_apple(self.dply, self.apple_pos)
            self.dply_snk(self.snk_pos)

            self.snk_pos, self.apple_pos, self.score = self.gnrt_snk(self.snk_hd,
                                                                     self.snk_pos,
                                                                     self.apple_pos,
                                                                     self.bttn_dir,
                                                                     self.score)
            pygame.display.set_caption("Snake" + "  " + "Score:  " + str(self.score))
            pygame.display.update()
            #self.prev_bttn_dir = self.bttn_dir
            """if self.drctn_blckd(self.snk_pos, self.crrnt_drctn_vctr) == 1:
                self.crash = True"""

            clock = pygame.time.Clock()
            clock.tick(20)
            return self.snk_pos, self.apple_pos, self.score

    def trngn_data(self):
        self.trng_data_x = []
        self.trng_data_y = []
        self.trng_games = 20
        self.steps_per_game = 500

        for _ in tqdm(range(self.trng_games)):
            self.snk_hd, self.snk_pos, self.apple_pos, self.score = self.starting_positions()
            self.prv_apple_dtnc = self.apple_dstnc_snk(self.apple_pos, self.snk_pos)
            self.prev_score = self.score

            for _ in range(self.steps_per_game):
                self.angle = self.apple_ang(self.snk_pos, self.apple_pos)
                self.direction, self.bttn_dir = self.nxt_direction(self.snk_pos, self.angle)
                self.snk_pos, self.apple_pos, self.score = self.game_loop(self.snk_hd,
                                                                          self.snk_pos, self.apple_pos,
                                                                          self.bttn_dir, self.score, self.apple_img)
                self.frnt_blckd, self.lft_blckd, self.rght_blckd = self.blckd_drctns(self.snk_pos)
                self.trng_data_x.append([self.lft_blckd, self.frnt_blckd,
                                         self.rght_blckd, self.angle, self.direction])
                if self.cllson_wth_edges(self.snk_pos[0]) == 1 or self.cllson_wth_snk(self.snk_pos) == 1:
                    self.trng_data_y.append(-1)
                    break
                else:
                    self.crrnt_snk_dstnc = self.apple_dstnc_snk(self.apple_pos, self.snk_pos)
                    if self.score > self.prev_score or self.crrnt_snk_dstnc < self.prv_apple_dtnc:
                        self.trng_data_y.append(1)
                    else:
                        self.trng_data_y.append(0)
                    self.prv_apple_dtnc = self.crrnt_snk_dstnc
                    self.prev_score = self.score

        return self.trng_data_x, self.trng_data_y

    def game_wth_NN(self, model):
        self.max_scr = 3
        self.avg_scr = 0
        self.test_games = 300
        self.test_steps = 300

        for _ in tqdm(range(self.test_games)):
            self.snk_hd, self.snk_pos, self.apple_pos, self.score = self.starting_positions()

            for _ in range(self.test_steps):
                self.frnt_blckd, self.lft_blckd, self.rght_blckd = self.blckd_drctns(
                    self.snk_pos)
                self.angle = self.apple_ang(self.snk_pos, self.apple_pos)
                self.predictions = []
                for i in range(-1, 2):
                    self.predictions.append(self.model.predict(
                        np.array([self.lft_blckd, self.frnt_blckd, self.rght_blckd, self.angle, i]).reshape(-1, 5, 1)))
                self.predicted_drctn = np.argmax(np.array(self.predictions)) - 1

                self.nxt_drctn = np.array(self.snk_pos[0]) - np.array(self.snk_pos[1])
                if self.predicted_drctn == -1:
                    self.nxt_drctn = np.array(
                        [self.nxt_drctn[1], -self.nxt_drctn[0]])
                if self.predicted_drctn == 1:
                    self.nxt_drctn = np.array(
                        [-self.nxt_drctn[1], self.nxt_drctn[0]])

                self.bttn_dir = self.gnrt_bttn_drctn(self.nxt_drctn)

                self.snk_pos, self.apple_pos, self.score = self.game_loop(
                    self.snk_hd, self.snk_pos, self.apple_pos, self.bttn_dir, self.score, self.apple_img)
                if self.cllson_wth_edges(self.snk_pos[0]) == 1 or self.cllson_wth_snk(self.snk_pos) == 1:
                    self.avg_scr += self.score
                    break

                if self.score > self.max_scr:
                    self.max_scr = self.score

        return self.max_scr, self.avg_scr/300

    def train_model(self):
        self.neural_network = input_data(shape=[None, 5, 1], name='input')
        self.neural_network = fully_connected(self.neural_network, 25, activation='relu')
        self.neural_network = fully_connected(self.neural_network, 10, activation='relu')
        self.neural_network = fully_connected(self.neural_network, 1, activation='tanh')
        self.neural_network = regression(
            self.neural_network, optimizer='adam', learning_rate=1e-3, loss='mean_square', name='target')
        self.model = tflearn.DNN(self.neural_network)

        return self.model

        # initialise game
    def on_init(self):
        pygame.init()
        self.dply = pygame.display.set_mode((self.dply_wdth, self.dply_hght))
        self.dply.fill(self.wndw_clr)
        pygame.display.update

        '''self.fnl_score = self.game_loop(self.snk_hd, self.snk_pos, self.apple_pos,
                                        1, self.apple_img, self.score)'''
        self.dply_txt = "Score: " + str(self.score)

        self.trng_data_x, self.trng_data_y = self.trngn_data()
        self.model = self.train_model()
        # print(self.trng_data_x)
        self.model.fit(np.array(self.trng_data_x).reshape(-1, 5, 1),
                       np.array(self.trng_data_y).reshape(-1, 1), n_epoch=3, shuffle=True)
        self.max_scr, self.avg_scr = self.game_wth_NN(self.model)


if __name__ == "__main__":
    app = snakeApp()
    app.on_init()
