library(ggplot2)
library(tidyverse)

plot_comparison_manual <- function() {
  
  models <- c('ResNet-18', 'ResNet-50', 'ResNeXt-50', 'ResNeXt-101', 'Inception-ResNet-v2')
  
  acc_2d <- c(0.93960084, 0.963060224, 0.967962185, 0.957107843, 0.969537815)
  acc_1d <- c(0.879726891, 0.93644958, 0.921918768, 0.881827731, 0.949054622)
  
  df <- data.frame(
    Model = models,
    `2D-CNN` = acc_2d * 100,
    `1D-CNN` = acc_1d * 100,
    check.names = FALSE
  )
  
  df_melted <- df %>%
    pivot_longer(cols = c(`2D-CNN`, `1D-CNN`),
                 names_to = "Version",
                 values_to = "Accuracy")
  
  df_melted$Model <- factor(df_melted$Model, levels = models)
  
  y_lower <- min(df_melted$Accuracy) - 5
  y_upper <- max(df_melted$Accuracy) + 3
  
  p <- ggplot(df_melted, aes(x = Model, y = Accuracy, fill = Version)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    geom_text(aes(label = sprintf("%.2f%%", Accuracy)),
              position = position_dodge(width = 0.7),
              vjust = -0.5,
              size = 3.5, fontface = "bold", family = "serif") +
    scale_fill_manual(values = c("2D-CNN" = "#1f77b4", "1D-CNN" = "#ff7f0e")) +
    coord_cartesian(ylim = c(y_lower, y_upper)) +
    labs(
      title = '1D-CNN vs. 2D-CNN Versions',
      y = 'Accuracy (%)',
      fill = 'Model Version'
    ) +
    theme_minimal() +
    theme(
      text = element_text(family = "serif"),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      legend.text = element_text(size = 12),
      legend.title = element_text(size = 12),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.y = element_blank()
    )
  
  print(p)
  ggsave("Figure_1D_vs_2D_Comparison_R.png", plot = p, width = 12, height = 7, dpi = 300)
}

plot_comparison_manual()
