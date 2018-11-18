import pygame
import random
import time
import numpy as np


class snakeApp:
    # initialise snake parameters
    dply_wdth = 550
    dply_hght = 550
    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 255)
    wndw_clr = (255, 255, 255)

    snk_hd = [250, 250]
    snk_pos = [[250, 250], [240, 250], [230, 250]]
    apple_pos = [random.randrange(1, 50)*10, random.randrange(1, 50)*10]
    score = 0
    apple_img = pygame.image.load('apple.jpg')

    def on_init(self):
        pygame.init()
        self.dply = pygame.display.set_mode((self.dply_wdth, self.dply_hght))
        self.dply.fill(self.wndw_clr)
        pygame.display.update

        self.fnl_score = self.game_loop(self.snk_hd, self.snk_pos, self.apple_pos,
                                        1, self.apple_img, self.score)
        self.dply_txt = "Score: " + str(self.score)

    def cllson_wth_apple(self, apple_pos, score):
        self.apple_pos = [random.randrange(1, 50)*10, random.randrange(1, 50)*10]
        self.score += 1
        return self.apple_pos, self.score

    def cllson_wth_edges(self, snk_hd):
        if self.snk_hd[0] >= self.dply_hght or self.snk_hd[0] \
                <= 0 or self.snk_hd[1] >= self.dply_hght or self.snk_hd[1] <= 0:
            return 1
        else:
            return 0

    def cllson_wth_snk(self, snk_pos):
        self.snk_hd = self.snk_pos[0]
        if self.snk_hd in snk_pos[2:]:
            return 1
        else:
            return 0

    def drctn_blckd(self, snk_pos, crrnt_drctn_vctr):
        self.step = self.snk_pos[0] + self.crrnt_drctn_vctr
        self.snk_hd = self.snk_pos[0]
        if self.cllson_wth_edges(self.snk_hd) == 1 or \
                self.cllson_wth_snk(self.snk_pos) == 1:
            return 1
        else:
            return 0

    def dply_snk(self, snk_pos):
        for position in self.snk_pos:
            pygame.draw.rect(self.dply, self.green,
                             pygame.Rect(position[0],
                                         position[1], 10, 10))

    def dply_apple(self, apple_pos, apple_img):
        self.dply.blit(self.apple_img, (self.apple_pos[0], self.apple_pos[1]))

    def gnrt_snk(self, snk_hd, snk_pos, apple_pos, bttn_dir, score):
        if self.bttn_dir == 1:
            self.snk_hd[0] += 10
        elif self.bttn_dir == 0:
            self.snk_hd[0] -= 10
        elif self.bttn_dir == 2:
            self.snk_hd[1] += 10
        elif self.bttn_dir == 3:
            self.snk_hd[1] -= 10
        else:
            pass

        if self.snk_hd == self.apple_pos:
            self.apple_pos, self.score = self.cllson_wth_apple(self.apple_pos, self.score)
            self.snk_pos.insert(0, list(self.snk_hd))

        else:
            self.snk_pos.insert(0, list(self.snk_hd))
            self.snk_pos.pop()
        return self.snk_pos, self.apple_pos, self.score

    def game_loop(self, snk_pos, snk_hd, apple, apple_pos, bttn_dir, score):
        self.crash = False
        self.prev_bttn_dir = 1
        self.bttn_dir = 1
        self.crrnt_drctn_vctr = np.array(self.snk_pos[0]) - np.array(self.snk_pos[1])
        while self.crash is not True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.crash = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and self.prev_bttn_dir != 1:
                        self.bttn_dir = 0
                    elif event.key == pygame.K_RIGHT and self.prev_bttn_dir != 0:
                        self.bttn_dir = 1
                    elif event.key == pygame.K_DOWN and self.prev_bttn_dir != 3:
                        self.bttn_dir = 2
                    elif event.key == pygame.K_UP and self.prev_bttn_dir != 2:
                        self.bttn_dir = 3
                    else:
                        self.bttn_dir = self.bttn_dir
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
            self.prev_bttn_dir = self.bttn_dir
            if self.drctn_blckd(self.snk_pos, self.crrnt_drctn_vctr) == 1:
                self.crash = True

            clock = pygame.time.Clock()
            clock.tick(15)
        return self.score


if __name__ == "__main__":
    app = snakeApp()
    app.on_init()
