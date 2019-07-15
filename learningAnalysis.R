library(tidyverse)
library(ggplot2)
library(gganimate)
library(magrittr)

df <- read.csv("./data/01_MF-hierENV.csv") %>%
    set_colnames(c("steps", "mu_steps")) %>%
    mutate(episode=c(1:nrow(.))) %>%
    mutate(env=rep("flat", nrow(.)))

dh <- read.csv("./data/01_MF-hierENV.csv") %>%
    set_colnames(c("steps", "mu_steps")) %>%
    mutate(episode=c(1:nrow(.))) %>%
    mutate(env=rep("hier", nrow(.)))

d <- rbind(df, dh)

p1 <- ggplot(d, aes(x=log(episode), y=steps, colour=env)) +
    geom_line() +
    transition_reveal(log(episode))

ggsave("./plots/MF.png", p1)
anim_save("./plots/MF.gif", p1))
