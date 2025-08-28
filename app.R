#---------------------------------------------------------------------------------------------#
#       Copyright (C) 2022, Author:  Laval Yannis Julien Jacquin                             #
#---------------------------------------------------------------------------------------------#
# This file is part of the Prio Retino+ software                                              #
#                                                                                             #
# Prio Retino+ software suite can be redistributed and/or modified under the terms of the     #
# GNU General Public License as published by the Free Software Foundation; either version 2   #
# of the License, or (at your option) any later version.                                      #
#                                                                                             #
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;   #
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.   #
# See the GNU General Public License for more details.                                        #
#                                                                                             #
# You should have received a copy of the GNU General Public License along with this program;  #
# if not, write to the Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,           #
# Boston, MA  02110-1301  USA                                                                 #
#---------------------------------------------------------------------------------------------#
options(rlib_downstream_check = FALSE)
library(devtools)
library(reticulate)
install_tensorflow_python_packages <- FALSE
if (install_tensorflow_python_packages) {
  install_miniconda(path = miniconda_path(), update = TRUE, force = FALSE)
  conda_create("prio_retino", python_version = "3.10")
  use_condaenv(condaenv = "prio_retino")
  library(tensorflow)
  install_tensorflow(version = "2.11.0", envname = "prio_retino")
  py_install("image-quality-1.2.7/", envname = "prio_retino", pip = TRUE)
  # if command above does not work use the following in shell
  # sudo /home/shiny/.local/share/r-miniconda/envs/prio_retino/bin/python -m pip install -e image-quality-1.2.7/
  py_install("matplotlib", envname = "prio_retino", pip = TRUE)
  py_install("opencv-python~=3.4.2", envname = "prio_retino", pip = TRUE)
  py_install("h5py~=3.7.0", envname = "prio_retino", pip = TRUE)
  py_config()
  py_available()
  py_module_available("cv2")
  py_module_available("imquality")
}
use_condaenv(condaenv = "prio_retino")
cv2 <- import("cv2")
imquality <- import("imquality")
library(tensorflow)
library(keras)
library(shiny)
library(shinyjs)
library(V8)
library(magick)
library(imager)
library(viridis)
library(data.table)
library(png)
library(stringr)
library(shinymanager)
library(shinycustomloader)
library(shiny.i18n)
library(shinyWidgets)
library(RSQLite)
library(dipsaus)
Sys.setenv("CUDA_VISIBLE_DEVICES" = "-1")
# library(rstudioapi)
# setwd(dirname(getActiveDocumentContext()$path))
options(encoding = "UTF-8")
source_python("img_resize_quality_functions.py")
source_python("grad_cam_functions.py")
# source custom function for concurrent writing in SQLite database
source("dbWriteTable_.R")

# set dbname
dbname_ <- "../user_data/prio_suite_db"

# user interface parameters

# set countries languages
countries <- c(
  "Français",
  "English",
  "Português"
)
flags <- c(
  "french.png",
  "us.png", 
  "portugal.png"
)

# file with translations
i18n <- Translator$new(translation_json_path = "prio_retino_translation.json")
i18n$set_translation_language("Français")

# create a connection to prio suite db and disconnect on exit
db_connect <- dbConnect(SQLite(), dbname = dbname_)
on.exit(DBI::dbDisconnect(db_connect))

# credentials
prio_retino_cred_use_df <- as.data.frame(dbReadTable(
  db_connect,
  "prio_retino_credential_current_month_usage"
))

# data.frame with credentials info
credentials <- data.frame(
  user = prio_retino_cred_use_df$Login,
  password = prio_retino_cred_use_df$Password,
  stringsAsFactors = FALSE
)

# full access list for all functionalities
full_access_list <- readLines("full_access_list")

# prio retino+ models and parameters

# cnn model and parameters
if (!exists("cnn_binary_classifier_0") && !exists("cnn_binary_classifier_1") && !exists("cnn_binary_classifier_2") &&
    !exists("cnn_binary_classifier_3") && !exists("cnn_binary_classifier_4") && !exists("cnn_binary_classifier_5")
) {
  cnn_binary_classifier_0 <<- load_model("xception_binary_classifier_0_full_arch_avg_pool_epochs_3.h5")
  cnn_binary_classifier_1 <<- load_model("xception_binary_classifier_1_full_arch_avg_pool_ratio_10_1_epochs_11.h5")
  cnn_binary_classifier_2 <<- load_model("xception_binary_classifier_2_full_arch_avg_pool_ratio_2_1_epochs_7.h5")
  cnn_binary_classifier_3 <<- load_model("xception_binary_classifier_3_full_arch_avg_pool_epochs_5.h5")
  cnn_binary_classifier_4 <<- load_model("xception_binary_classifier_4_full_arch_avg_pool_epochs_5.h5")
  cnn_binary_classifier_5 <<- load_model("xception_binary_classifier_5_full_arch_avg_pool_ratio_10_1_epochs_7.h5")
}
img_size_cnn <<- 299

# raw image parameters
desired_size <<- 1024
img_size_qual <<- 150
blur_factor <<- 100
img_qual_tresh <<- 32

# output image parameters
width_img_size <<- 540
height_img_size <<- 450

# grad-cam parameter
last_conv_layer_name <<- "block14_sepconv2_act"

# convert output images to low quality to increase rendering speed
annotation_color_ <<- "none" # render in white

# loading css content
appCSS <- "
#loading-content {
  position: absolute;
  background: #000000;
  opacity: 0.9;
  z-index: 100;
  left: 0;
  right: 0;
  height: 100%;
  text-align: center;
  color: #FFFFFF;
}
"

# user interface components
ui <- secure_app(
  choose_language = FALSE,
  tags_top = tags$img(src = "logo_prio_retino_plus.png", width = 300),
  
  # ui <- fluidPage(
  fluidPage(
    HTML('<meta name="viewport" content="width=1024">'),
    useShinyjs(),
    inlineCSS(appCSS),
    
    # loading message
    div(
      id = "loading-content",
      h2(i18n$t("Chargement de l'application Prio Retino+..."))
    ),
    
    # language selection
    shiny.i18n::usei18n(i18n),
    div(
      style = "float: right;", class = "chooselang",
      pickerInput(
        inputId = "selected_language",
        label = i18n$t("Définir la langue de Prio Retino+"),
        choices = i18n$get_languages(),
        selected = i18n$get_key_translation(),
        choicesOpt = list(
          content =
            mapply(countries, flags, FUN = function(country, flagUrl) {
              HTML(paste(
                tags$img(src = flagUrl, width = 20, height = 15),
                country
              ))
            }, SIMPLIFY = FALSE, USE.NAMES = FALSE)
        )
      )
    ),
    
    # main app code goes here
    div(
      id = "app-content",
      titlePanel("", windowTitle = "Prio Retino+"),
      sidebarLayout(
        sidebarPanel(
          HTML('<center><img src="logo_moju_ai.png" width="180"></header>'),
          column(width = 1, offset = 10, style = "padding:6px;"),
          #
          titlePanel(h5(p(strong(i18n$t("Ne pas rafraichir la page, appuyer sur 'Réinitialiser' pour nettoyer avant chaque nouvelle analyse"))),
                        style = "color:red", align = "left"
          )),
          actionButton("reset", i18n$t("Réinitialiser")),
          titlePanel(h5(p(""), align = "left")),
          #
          textInput("patient_id", i18n$t("Insérer l'identifiant du patient"), value = ""),
          fancyFileInput(
            input = "file1",
            label = i18n$t("Charger l'image du fond d'oeil"),
            accept = c(".png", ".jpeg", ".jpg")
          ),
          uiOutput("UIselectInput"),
          verbatimTextOutput("res_auth"),
          width = 3
        ),
        mainPanel(
          tags$style(
            type = "text/css", ".shiny-output-error { visibility: hidden; }",
            ".shiny-output-error:before { visibility: hidden; }"
          ),
          tags$head(
            tags$style("#text1{color: blue;
                             font-size: 50px;
                             font-style: italic;
                             }")
          ),
          conditionalPanel(
            'input.element_id=="Diabetic retinopathy and/or maculopathy"||
               input.element_id=="Rétinopathie et/ou maculopathie diabétique"||
               input.element_id=="Retinopatia diabética e/ou maculopatia"',
            h4(htmlOutput("output_dr_prio_retino_text"), align = "left")
          ),
          conditionalPanel(
            'input.element_id=="Glaucoma"||
               input.element_id=="Glaucome"||
               input.element_id=="Glaucoma" ',
            h4(htmlOutput("output_glauco_prio_retino_text"), align = "left")
          ),
          withLoader(imageOutput("outputImage"),
                     type = "image", loader = "computation_loader_new.gif"
          )
        )
      ),
      useShinyjs(),
      extendShinyjs(text = "shinyjs.winprint = function(){ window.print(); }", functions = c("winprint")),
      actionButton("print", i18n$t("Imprimer ou sauvegarder les résultats de Prio Retino+ au format PDF")),
      titlePanel(h6(p(strong(i18n$t("Remarques, définitions et recommandations :"))), style = "color:#0c7683", align = "left")),
      titlePanel(h6(i18n$t("- Les résultats de Prio Retino+ sont donnés à titre indicatif, le diagnostic doit être établi par un ophtalmologue"),
                    style = "color:#0c7683", align = "left"
      )),
      titlePanel(h6(i18n$t("- Les données analysées sont supprimées et ne sont pas stockées par Prio Retino+ après chaque réinitialisation"),
                    style = "color:#0c7683", align = "left"
      )),
      titlePanel(h6(i18n$t("- EIRD : Echelle Internationale de sévérité de la Rétinopathie Diabétique; AAO : Académie Américaine d'Ophtalmologie"),
                    style = "color:#0c7683", align = "left"
      )),
      titlePanel(h6(i18n$t("- Rétinopathie diabétique (RD) non référable : léger ou aucun signe visible de la RD selon l'EIRD. Recommandations de l'AAO : répéter l'examen annuellement pour la RD non référable"),
                    style = "color:#0c7683", align = "left"
      )),
      titlePanel(h6(i18n$t("- Rétinopathie diabétique (RD) référable : signes modérés ou supérieurs de la RD selon l'EIRD. Recommandations de l'AAO : répéter l'examen respectivement tous les 6 ou 3 mois pour des signes modérés ou supérieurs de RD"),
                    style = "color:#0c7683", align = "left"
      )),
      titlePanel(h6(i18n$t("- Glaucome non référable : aucun signe visible ou faible risque suspecté de glaucome. Recommandations de l'AAO : répéter l'examen chaque année"),
                    style = "color:#0c7683", align = "left"
      )),
      titlePanel(h6(i18n$t("- Glaucome référable : vrai glaucome, glaucome pré-périmétrique ou haut risque suspecté. Recommandations de l'AAO : répéter l'examen tous les 1 à 2 mois jusqu'à stabilisation de la maladie"),
                    style = "color:#0c7683", align = "left"
      ))
    ),
    
    # global JS for capturing pasted image
    # tags$script(HTML("
    #   document.addEventListener('paste', function(e) {
    #     var items = (e.clipboardData || e.originalEvent.clipboardData).items;
    #     for (var i = 0; i < items.length; i++) {
    #       if (items[i].type.indexOf('image') !== -1) {
    #         var blob = items[i].getAsFile();
    #         var reader = new FileReader();
    #         reader.onload = function(event){
    #           Shiny.setInputValue('pasted_image', event.target.result, {priority: 'event'});
    #         };
    #         reader.readAsDataURL(blob);
    #       }
    #     }
    #   });
    # "))
  )
)

# server component
server <- shinyServer(
  function(input, output, session) {
    options(shiny.maxRequestSize = 4 * 1024^2)
    
    # check credentials
    result_auth <- secure_server(check_credentials = check_credentials(credentials))
    output$res_auth <- renderPrint({
      reactiveValuesToList(result_auth)
    })
    
    # hide the loading message when the reset of the server function has executed
    hide(id = "loading-content", anim = TRUE, animType = "fade", time = 4)
    
    # create reactive values for input file and patient id
    rv <- reactiveValues(
      file1 = NULL,
      patient_id = NULL,
    )
    
    # reset input file and patient id
    observeEvent(input$reset, {
      rv$file1 <- NULL
      rv$patient_id <- NULL
      reset("file1")
      reset("patient_id")
    })
    
    # set patient id during non analysis state only
    observeEvent(input$patient_id, {
      non_analysis_state <- (list_out_prio_retino()$out_dr_prio_retino_txt == "")
      if (non_analysis_state
      ) {
        rv$patient_id <- input$patient_id
      }
    })
    
    # set input file during non analysis state only
    observeEvent(input$file1, {
      non_analysis_state <- (list_out_prio_retino()$out_dr_prio_retino_txt == "")
      if (non_analysis_state
      ) {
        rv$file1 <- input$file1
      }
    })
    
    # file input via CTRL+V (clipboard image)
    # observeEvent(input$pasted_image, {
    #   non_analysis_state <- (list_out_prio_retino()$out_dr_prio_retino_txt == "")
    #   if (non_analysis_state
    #   ) {
    #     tmpF <- tempfile(fileext = ".png")
    #     base64_data <- sub("^data:image/.+;base64,", "", input$pasted_image)
    #     writeBin(base64enc::base64decode(base64_data), tmpF)
    #     rv$file1 <- list(datapath = tmpF, size = file.info(tmpF)$size)
    #   }
    # })
    
    # set language during non analysis state only
    observeEvent(input$selected_language, {
      non_analysis_state <- (list_out_prio_retino()$out_dr_prio_retino_txt == "")
      if (non_analysis_state
      ) {
        # here is where we update language in session
        shiny.i18n::update_lang(session, input$selected_language)
      }
    })
    
    # translate analyzed diseases
    output$UIselectInput <- renderUI({
      auth_ind <- as.character(reactiveValuesToList(result_auth))
      if (auth_ind %in% full_access_list) {
        selectInput("element_id",
                    label = i18n$t("Afficher les résultats de pré-diagnostic pour :"),
                    choices = i18n$t(c(
                      "Rétinopathie et/ou maculopathie diabétique",
                      "Glaucome"
                    ))
        )
      } else {
        selectInput("element_id",
                    label = i18n$t("Afficher les résultats de pré-diagnostic pour :"),
                    choices = i18n$t(
                      "Rétinopathie et/ou maculopathie diabétique"
                    )
        )
      }
    })
    
    # print results
    observeEvent(input$print, {
      js$winprint()
    })
    
    # make prio computations and return results as a list
    list_out_prio_retino <- reactive({
      # initialize an empty list for prio retino results
      list_out_prio_retino <- list(
        pred_other_img_status = NULL,
        resized_cropped_target_image = NULL,
        transformed_target_image = NULL,
        resized_transformed_target_image = NULL,
        out_orig_dr_img = NULL,
        out_orig_glauco_img = NULL,
        out_dr_prio_retino_txt = NULL,
        out_glauco_prio_retino_txt = NULL
      )
      
      if (!is.null(unlist(rv$file1)) && as.numeric(rv$file1$size) > 1) {
        # test if the loaded file is a fundus image
        img <- image_load(rv$file1$datapath, target_size = c(img_size_cnn, img_size_cnn))
        x <- image_to_array(img)
        x <- array_reshape(x, c(1, dim(x)))
        x <- x / 255
        proba_other_img_status <- as.numeric(cnn_binary_classifier_0 %>% predict(x))
        ifelse((proba_other_img_status <= 0.5), pred_other_img_status <- 0, pred_other_img_status <- 1)
        list_out_prio_retino$pred_other_img_status <- pred_other_img_status
        
        if ((!is.null(unlist(rv$patient_id))) && (unlist(rv$patient_id) != "") && (!pred_other_img_status)) {
          # create a connection to prio suite db and disconnect on exit
          db_connect <- dbConnect(SQLite(), dbname = dbname_)
          on.exit(DBI::dbDisconnect(db_connect))
          
          # update login usage
          auth_ind <- as.character(reactiveValuesToList(result_auth))
          prio_retino_cred_use_df <- as.data.frame(dbReadTable(
            db_connect,
            "prio_retino_credential_current_month_usage"
          ))
          current_count <- prio_retino_cred_use_df[match(auth_ind, prio_retino_cred_use_df$Login), ]$Count
          prio_retino_cred_use_df[match(auth_ind, prio_retino_cred_use_df$Login), ]$Count <- current_count + 1
          prio_retino_cred_use_df[match(auth_ind, prio_retino_cred_use_df$Login), ]$Last_analysis_timestamp <- paste0(as.character(Sys.time()), "sec")
          dbWriteTable_(db_connect, "prio_retino_credential_current_month_usage",
                        prio_retino_cred_use_df,
                        overwrite_ = TRUE
          )
          # resize and crop original target image
          out_resize_qual <- compute_resize_quality_img(filename = rv$file1$datapath, img_size_qual = img_size_qual, desired_size = desired_size)
          list_out_prio_retino$resized_cropped_target_image <- image_read(out_resize_qual$temp_resize_img)
          file.remove(out_resize_qual$temp_resize_img)
          
          # convert original image to transformed target image
          transformed_target_image <- image_scale(list_out_prio_retino$resized_cropped_target_image, desired_size)
          transformed_target_image <- image_resize(transformed_target_image, desired_size)
          transformed_target_image <- magick2cimg(transformed_target_image)
          Blur_target_img <- boxblur(transformed_target_image, blur_factor)
          transformed_target_image <- transformed_target_image - Blur_target_img
          
          # save and resize transformed target image
          tmpF_trans_img <- tempfile(fileext = ".png")
          save.image(
            transformed_target_image,
            tmpF_trans_img,
            quality = 1
          )
          list_out_prio_retino$transformed_target_image <- image_load(tmpF_trans_img)
          list_out_prio_retino$resized_transformed_target_image <- image_load(tmpF_trans_img,
                                                                              target_size = c(img_size_cnn, img_size_cnn)
          )
          
          # convert transformed image to array
          x_target <- image_to_array(list_out_prio_retino$resized_transformed_target_image)
          x_target <- array_reshape(x_target, c(1, dim(x_target)))
          x_target <- x_target / 255
          
          # make an output for original image
          out_orig_img <- image_scale(list_out_prio_retino$resized_cropped_target_image, "412x412!") %>%
            image_annotate(i18n$t("Fond d'oeil original"),
                           font = "monospace",
                           color = "white", size = 12
            )
          
          # compute dr status
          proba_dr_status <- as.numeric(cnn_binary_classifier_1 %>% predict(x_target))
          ifelse((proba_dr_status <= 0.5), pred_dr_status <- 0, pred_dr_status <- 1)
          
          if (pred_dr_status == 0) {
            # make an img output for no dr
            out_dr_img <- image_scale(magick::image_read(
              image_array_resize(list_out_prio_retino$transformed_target_image,
                                 height = img_size_cnn, width = img_size_cnn
              ) / 255
            ), "412x412!")
            
            # make a txt output for no dr
            out_dr_txt <- paste0(
              i18n$t("Résultats de Prio Retino+ : léger ou aucun signe visible de rétinopathie diabétique (i.e. RD non référable) détecté avec une probabilité de "),
              (1 - trunc(100 * proba_dr_status) / 100), i18n$t(" pour "), rv$patient_id, "."
            )
            dr_color <- "#49DC67"
            
            # make a txt and img output for no dr and maculo
            proba_maculo <- 1 - as.numeric(cnn_binary_classifier_3 %>% predict(x_target))
            ifelse((proba_maculo <= 0.5), Maculo_status <- 0, Maculo_status <- 1)
            if (Maculo_status) {
              out_dr_txt <- paste0(
                out_dr_txt, i18n$t("Attention : présence possible de maculopathie détectée avec une probabilité de "),
                (trunc(100 * proba_maculo) / 100)
              )
              dr_color <- "#FF5050"
              out_dr_img <- out_dr_img %>% image_annotate(i18n$t("Zones détectées pour la maculopathie diabétique"),
                                                          font = "monospace", color = annotation_color_, size = 12
              )
            } else {
              out_dr_img <- out_dr_img %>% image_annotate(i18n$t("Aucune zone détectée pour la rétinopathie \n et/ou maculopathie diabétique"),
                                                          font = "monospace", color = annotation_color_, size = 12
              )
            }
          } else {
            proba_dr_level <- as.numeric(cnn_binary_classifier_2 %>% predict(x_target))
            ifelse((proba_dr_level <= 0.5), pred_dr_level <- 0, pred_dr_level <- 1)
            
            proba_maculo <- 1 - as.numeric(cnn_binary_classifier_4 %>% predict(x_target))
            ifelse((proba_maculo <= 0.5), Maculo_status <- 0, Maculo_status <- 1)
            
            ifelse((pred_dr_level == 0),
                   dr_level <- paste0(
                     i18n$t("signes potentiels de RD modérée détectés avec une probabilité de "),
                     (1 - trunc(100 * proba_dr_level) / 100)
                   ),
                   dr_level <- paste0(
                     i18n$t("signes potentiels de RD sévère ou supérieure détectés avec une probabilité de "),
                     (trunc(100 * proba_dr_level) / 100)
                   )
            )
            out_dr_txt <- paste0(
              i18n$t("Résultats de Prio Retino+ : rétinopathie diabétique référable détectée avec une probabilité de "), (trunc(100 * proba_dr_status) / 100),
              i18n$t(" pour "), rv$patient_id, ".", i18n$t(" Stade de la maladie :  "), dr_level, ". "
            )
            if (Maculo_status) {
              out_dr_txt <- paste0(
                out_dr_txt, i18n$t("Attention : présence possible de maculopathie détectée avec une probabilité de "),
                (trunc(100 * proba_maculo) / 100)
              )
            }
            dr_color <- c("#FF5050", "#FF5050")[pred_dr_level + 1]
            
            # compute grad classification activation mapping for dr status
            out_dr_img <- image_scale(magick::image_read(
              compute_grad_cam(
                cnn_binary_classifier_1,
                list_out_prio_retino$transformed_target_image,
                list_out_prio_retino$resized_transformed_target_image,
                last_conv_layer_name
              ) / 255
            ), "412x412!") %>% image_annotate(i18n$t("Zones détectées pour la rétinopathie et/ou maculopathie diabétique"),
                                              font = "monospace", color = annotation_color_, size = 12
            )
          } # end else for pred_dr_status test
          list_out_prio_retino$out_orig_dr_img <- image_append(c(out_orig_img, out_dr_img))
          list_out_prio_retino$out_orig_dr_img <- magick2cimg(list_out_prio_retino$out_orig_dr_img,
                                                              alpha = "flatten"
          )
          
          # compute glauco status
          proba_glauco_status <- as.numeric(cnn_binary_classifier_5 %>% predict(x_target))
          
          ifelse((proba_glauco_status <= 0.5), glauco_status <- 0, glauco_status <- 1)
          if (glauco_status == 0) {
            out_glauco_txt <- paste0(
              i18n$t("Résultats de Prio Retino+ : aucun signe visible ou faible risque suspecté de glaucome (i.e. glaucome non référable) détecté avec une probabilité de "),
              (1 - trunc(100 * proba_glauco_status) / 100), i18n$t(" pour "), rv$patient_id, "."
            )
            glauco_color <- "#49DC67"
            # make an img output for no glauco
            out_glauco_img <- image_scale(magick::image_read(
              image_array_resize(list_out_prio_retino$transformed_target_image,
                                 height = img_size_cnn, width = img_size_cnn
              ) / 255
            ), "412x412!") %>% image_annotate(i18n$t("Aucune zone détectée pour le glaucome"),
                                              font = "monospace", color = annotation_color_, size = 12
            )
          } else {
            out_glauco_txt <- paste0(
              i18n$t("Résultats de Prio Retino+ : glaucome référable (i.e. vrai glaucome, glaucome pré-périmétrique ou haut risque suspecté) détecté avec une probabilité de "),
              trunc(100 * proba_glauco_status) / 100, i18n$t(" pour "), rv$patient_id, "."
            )
            glauco_color <- "#FF5050"
            # compute grad classification activation mapping for glauco status
            out_glauco_img <- image_scale(magick::image_read(
              compute_grad_cam(
                cnn_binary_classifier_5,
                list_out_prio_retino$transformed_target_image,
                list_out_prio_retino$resized_transformed_target_image,
                last_conv_layer_name
              ) / 255
            ), "412x412!") %>% image_annotate(i18n$t("Zones détectées pour le glaucome"),
                                              font = "monospace", color = annotation_color_, size = 12
            )
          } # end glauco status computation
          list_out_prio_retino$out_orig_glauco_img <- image_append(c(out_orig_img, out_glauco_img))
          list_out_prio_retino$out_orig_glauco_img <- magick2cimg(list_out_prio_retino$out_orig_glauco_img,
                                                                  alpha = "flatten"
          )
          
          # test image quality and add warning
          ifelse(as.numeric(out_resize_qual$img_qual_score) < img_qual_tresh, img_qual <- 1, img_qual <- 0)
          if (!img_qual) {
            dr_color <- "#EE9F27"
            glauco_color <- "#EE9F27"
            img_qual_warning <- i18n$t("Attention : image de faible qualité détectée, les résultats de Prio Retino+ peuvent ne pas être fiables. ")
            out_dr_txt <- paste0(img_qual_warning, out_dr_txt)
            out_glauco_txt <- paste0(img_qual_warning, out_glauco_txt)
          }
          
          out_dr_prio_retino_txt <- HTML(paste0("<div style='background-color:", dr_color, "'>", out_dr_txt, "</div>"))
          out_glauco_prio_retino_txt <- HTML(paste0("<div style='background-color:", glauco_color, "'>", out_glauco_txt, "</div>"))
        } else {
          if (!pred_other_img_status) {
            out_dr_prio_retino_txt <- HTML(paste0("<font color='#0c7683'>", i18n$t("Veuillez d'abord réinitialiser Prio Retino+, puis suivez ces instructions : 1. Sélectionnez votre langue, 2. Insérez l'identifiant du patient et 3. Chargez une image du fond d'oeil."), "</font>"))
            out_glauco_prio_retino_txt <- HTML(paste0("<font color='#0c7683'>", i18n$t("Veuillez d'abord réinitialiser Prio Retino+, puis suivez ces instructions : 1. Sélectionnez votre langue, 2. Insérez l'identifiant du patient et 3. Chargez une image du fond d'oeil."), "</font>"))
          } else {
            out_dr_prio_retino_txt <- HTML(paste0("<font color='#0c7683'>", paste0(i18n$t("Le fichier téléchargé est-il une image de fond d'oeil ? "), i18n$t("Veuillez d'abord réinitialiser Prio Retino+, puis suivez ces instructions : 1. Sélectionnez votre langue, 2. Insérez l'identifiant du patient et 3. Chargez une image du fond d'oeil.")), "</font>"))
            out_glauco_prio_retino_txt <- HTML(paste0("<font color='#0c7683'>", paste0(i18n$t("Le fichier téléchargé est-il une image de fond d'oeil ? "), i18n$t("Veuillez d'abord réinitialiser Prio Retino+, puis suivez ces instructions : 1. Sélectionnez votre langue, 2. Insérez l'identifiant du patient et 3. Chargez une image du fond d'oeil.")), "</font>"))
          }
        }
        list_out_prio_retino$out_dr_prio_retino_txt <- out_dr_prio_retino_txt
        list_out_prio_retino$out_glauco_prio_retino_txt <- out_glauco_prio_retino_txt
      } else {
        list_out_prio_retino$out_dr_prio_retino_txt <- ""
        list_out_prio_retino$out_glauco_prio_retino_txt <- ""
      }
      list_out_prio_retino
    })
    
    # output image
    output$outputImage <- renderImage(
      {
        list_out_prio_retino <- list_out_prio_retino()
        test_render_img <- (!is.null(unlist(rv$file1)) && as.numeric(rv$file1$size) > 1) &&
          ((!is.null(unlist(rv$patient_id))) && (unlist(rv$patient_id) != "")) &&
          (!list_out_prio_retino$pred_other_img_status)
        
        if (!test_render_img) {
          # default image
          list(
            src = "www/ai_prio_retino_plus.png", contentType = "image/png",
            width = 1320, height = 580, align = "left"
          )
        } else {
          # return images for DR and/or maculopathy
          if (input$element_id == "Diabetic retinopathy and/or maculopathy" ||
              input$element_id == "Rétinopathie et/ou maculopathie diabétique" ||
              input$element_id == "Retinopatia diabética e/ou maculopatia") {
            tmpF_orig_dr_img <- tempfile(fileext = ".png")
            save.image(
              list_out_prio_retino$out_orig_dr_img,
              tmpF_orig_dr_img,
              quality = 1
            )
            list(
              src = tmpF_orig_dr_img,
              contentType = "image/png",
              width = 1200, height = 550
            )
            # return images for glaucoma
          } else {
            tmpF_orig_glauco_img <- tempfile(fileext = ".png")
            save.image(
              list_out_prio_retino$out_orig_glauco_img,
              tmpF_orig_glauco_img,
              quality = 1
            )
            list(
              src = tmpF_orig_glauco_img,
              contentType = "image/png",
              width = 1200, height = 550
            )
          }
        }
      },
      deleteFile = FALSE
    )
    
    output$output_dr_prio_retino_text <- renderText({
      if (!(!is.null(unlist(rv$file1)) && as.numeric(rv$file1$size) > 1)) {
        ""
      } else if (input$element_id == "Diabetic retinopathy and/or maculopathy" ||
                 input$element_id == "Rétinopathie et/ou maculopathie diabétique" ||
                 input$element_id == "Retinopatia diabética e/ou maculopatia") {
        list_out_prio_retino()$out_dr_prio_retino_txt
      }
    })
    
    output$output_glauco_prio_retino_text <- renderText({
      if (!(!is.null(unlist(rv$file1)) && as.numeric(rv$file1$size) > 1)) {
        ""
      } else if (input$element_id == "Glaucoma" ||
                 input$element_id == "Glaucome" ||
                 input$element_id == "Glaucoma") {
        list_out_prio_retino()$out_glauco_prio_retino_txt
      }
    })
  }
)

shinyApp(ui, server)
