/*****************************************************************

Author: Albert Nadal Garriga
Date: 23/01/2022
Description: Simple exploratory tool that uses a Self-Organizing Map (SOM algorithm) to reduce the dimensionality
             of the Wine Quality dataset (https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)

Usage:
1) Install Raylib dependency on Linux:
git clone https://github.com/raysan5/raylib.git raylib
sudo apt install libasound2-dev mesa-common-dev libx11-dev libxrandr-dev libxi-dev xorg-dev libgl1-mesa-dev
cd raylib/src/
make PLATFORM=PLATFORM_DESKTOP
sudo make install

2) Compile and run the application:
gcc -std=c99 -Wno-unused-result -O3 som.c -o som -lm -lraylib -pthread -ldl
./som

Notes: This is just a POC implementation that needs some refactoring. This software is intended to be used for
       educational purposes. Please, feel free to use or improve this code.

*****************************************************************/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <string.h>
#include <raylib.h>

// Window size
#define SCREEN_WIDTH 1200
#define SCREEN_HEIGHT 1200

// Neural Network size
#define MAP_WIDTH 300
#define MAP_HEIGHT 300

// Neural Network canvas size
#define MAP_LAYOUT_WIDTH 900
#define MAP_LAYOUT_HEIGHT 900

// Training algorithm parameters
#define INITIAL_TRAINING_ITERATIONS_PER_EPOCH 300
#define TOTAL_EPOCHS 8
#define INITIAL_RADIUS 200.0L
#define INITIAL_LEARNING_RULE 0.9L

// Basic math macros
#define pow2(x) ((x) * (x))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(X, Y) (((X) < (Y)) ? (X) : (Y))

// Total colors paintbrush palette
#define MAX_COLORS_COUNT 18

typedef struct Neuron
{
  double *weights;
} Neuron;

typedef struct BMU
{
  int x_coord;
  int y_coord;
} BMU;

typedef struct Coordinate
{
  double x;
  double y;
} Coordinate;

typedef struct Sample
{
  double *components;
  double value;
  BMU bmu;
} Sample;

typedef struct ComponentInfo
{
  char *name;
  double max_value;
  char *max_value_str;
  double min_value;
  char *min_value_str;
} ComponentInfo;

typedef struct DatasetInfo
{
  ComponentInfo *components;
  Sample *samples;
  int total_components;
  int total_dataset_samples;
} DatasetInfo;

Neuron **map;
DatasetInfo info = {
    .components = NULL,
    .samples = NULL,
    .total_components = 0,
    .total_dataset_samples = 0};

char dataset_csv_file[] = "winequality-white-normalized.csv";
int epoch = 0;
int iteration = 0;
int iterations_per_epoch = INITIAL_TRAINING_ITERATIONS_PER_EPOCH;

int get_total_ocurrences_of_char(const char *s, char c)
{
  int i, count = 0;
  for (i = 0; s[i]; i++)
    if (s[i] == c)
      count++;
  return count;
}

int get_csv_total_rows_from_file(char *filename)
{
  int count = 0;
  FILE *fp = fopen(filename, "r");

  if (fp == NULL)
  {
    printf("Could not open file %s", filename);
    return 0;
  }

  for (char c = getc(fp); c != EOF; c = getc(fp))
    if (c == '\n')
      count++;

  fclose(fp);

  if (count >= 3)
    count -= 3; // Discount the header row, min values row, max values row

  return count;
}

void load_dataset_info(char *filename)
{
  FILE *fp = fopen(filename, "r");
  char buffer[512];
  char *ptr;

  if (fp == NULL)
  {
    printf("Could not open file %s", filename);
    return;
  }

  // Read the first line and get the fields names
  fgets(buffer, sizeof(buffer), fp);

  info.total_components = get_total_ocurrences_of_char(buffer, ';') + 1;
  printf("Total fields: %d\n", info.total_components);
  info.components = malloc(info.total_components * sizeof(ComponentInfo));

  char *token = strtok(buffer, ";");
  int i = 0;
  while (token != NULL)
  {
    info.components[i].name = malloc((strlen(token) + 1) * sizeof(char));
    printf(" field %d: %s\n", i, token);
    strcpy(info.components[i++].name, token);
    token = strtok(NULL, ";");
  }

  // Read the second line and get the min value of each component
  fgets(buffer, sizeof(buffer), fp);
  token = strtok(buffer, ";");
  i = 0;
  while (token != NULL)
  {
    info.components[i].min_value_str = malloc((strlen(token) + 1) * sizeof(char));
    info.components[i].min_value = strtod(token, &ptr);
    strcpy(info.components[i++].min_value_str, token);
    token = strtok(NULL, ";");
  }

  /// Read the third line and get the max value of each component
  fgets(buffer, sizeof(buffer), fp);
  fclose(fp);

  token = strtok(buffer, ";");
  i = 0;
  while (token != NULL)
  {
    info.components[i].max_value_str = malloc((strlen(token) + 1) * sizeof(char));
    info.components[i].max_value = strtod(token, &ptr);
    strcpy(info.components[i++].max_value_str, token);
    token = strtok(NULL, ";");
  }

  info.total_dataset_samples = get_csv_total_rows_from_file(filename);
  printf("\n\nTotal samples: %d\n\n", info.total_dataset_samples);
}

void load_dataset_samples(char *filename)
{
  // 'samples' is the data structure used to store the sample points for Kohonen algorithm process
  info.samples = (Sample *)malloc(sizeof(Sample) * info.total_dataset_samples);

  for (int i = 0; i < info.total_dataset_samples; i++)
  {
    info.samples[i].components = (double *)malloc(sizeof(double) * (info.total_components - 1));
    info.samples[i].bmu.x_coord = 0;
    info.samples[i].bmu.y_coord = 0;
  }

  FILE *fp = fopen(filename, "r");
  char buffer[512];
  int i, line_count = 0;
  char *end_ptr;

  if (fp == NULL)
  {
    printf("Could not open file %s", filename);
    return;
  }

  // ignore the first 3 lines
  for (int e = 0; e < 3; e++)
    fgets(buffer, sizeof(buffer), fp);

  while (fgets(buffer, sizeof(buffer), fp) && (line_count < info.total_dataset_samples))
  {
    char *token = strtok(buffer, ";");
    i = 0;
    while (token != NULL)
    {
      if (i == info.total_components - 1)
        info.samples[line_count].value = strtod(token, &end_ptr);
      else
        info.samples[line_count].components[i] = strtod(token, &end_ptr);
      token = strtok(NULL, ";");
      i++;
    }
    line_count++;
  }

  fclose(fp);
}

void load_dataset(char *filename)
{
  // Load dataset attributes names, total attributes and total samples
  load_dataset_info(filename);

  // Load dataset samples
  load_dataset_samples(filename);
}

void initialize_som_map()
{
  map = (Neuron **)malloc(sizeof(Neuron *) * MAP_WIDTH);

  for (int x = 0; x < MAP_WIDTH; x++)
    map[x] = (Neuron *)malloc(sizeof(Neuron) * MAP_HEIGHT);

  for (int x = 0; x < MAP_WIDTH; x++)
    for (int y = 0; y < MAP_HEIGHT; y++)
    {
      map[x][y].weights = (double *)malloc(sizeof(double) * (info.total_components - 1));
      for (int i = 0; i < info.total_components - 1; i++)
        map[x][y].weights[i] = (double)rand() / (double)RAND_MAX; // a random double value between 0 and 1
    }
}

Sample *pick_random_sample()
{
  int i = rand() % info.total_dataset_samples;
  return &info.samples[i];
}

double distance_between_sample_and_neuron(Sample *sample, Neuron *neuron, int total_components)
{
  double euclidean_distance = 0.0f;
  double component_diff;

  for (int i = 0; i < total_components - 1; i++)
  {
    component_diff = sample->components[i] - neuron->weights[i];
    euclidean_distance += pow2(component_diff);
  }

  //return the Euclidean_distance;
  return sqrt(euclidean_distance);
}

void search_bmu(Sample *sample, BMU *bmu, int total_components)
{
  double dist, min_dist = DBL_MAX;
  for (int x = 0; x < MAP_WIDTH; x++)
    for (int y = 0; y < MAP_HEIGHT; y++)
    {
      dist = distance_between_sample_and_neuron(sample, &map[x][y], total_components);
      if (dist < min_dist)
      {
        bmu->x_coord = x;
        bmu->y_coord = y;
        min_dist = dist;
      }
    }
}

double get_coordinate_distance(Coordinate *p1, Coordinate *p2)
{
  double x_sub = (p1->x) - (p2->x);
  double y_sub = (p1->y) - (p2->y);
  return sqrt(x_sub * x_sub + y_sub * y_sub);
}

void scale_neuron_at_position(int x, int y, Sample *sample, double scale, int total_components)
{
  double neuron_prescaled, neuron_scaled;
  Neuron *neuron = &map[x][y];

  for (int i = 0; i < total_components - 1; i++)
  {
    neuron_prescaled = neuron->weights[i] * (1.0f - scale);
    neuron_scaled = (sample->components[i] * scale) + neuron_prescaled;
    neuron->weights[i] = (double)neuron_scaled;
  }
}

void scale_neighbors(BMU *bmu, Sample *sample, double iteration_radius, double learning_rule, int total_components)
{
  static Coordinate center = {
          .x = 0.0L,
          .y = 0.0L
  };

  static Coordinate outer;
  outer.x = outer.y = iteration_radius;

  double distance, scale;
  int y_coord, x_coord, x_offset, y_offset, int_iteration_radius = (int)iteration_radius;

  for (int y = -int_iteration_radius; y < int_iteration_radius; y++)
    for (int x = -int_iteration_radius; x < int_iteration_radius; x++)
      {
        outer.x = x;
        outer.y = y;
        distance = get_coordinate_distance(&outer, &center);
        if (distance < iteration_radius)
        {
          scale = learning_rule * exp(-10.0f * (distance * distance) / (iteration_radius * iteration_radius));
          x_offset = x + bmu->x_coord;
          y_offset = y + bmu->y_coord;
          x_coord = x_offset < 0 ? MAP_WIDTH + x_offset : (x_offset >= MAP_WIDTH ? x_offset - MAP_WIDTH: x_offset);
          y_coord = y_offset < 0 ? MAP_HEIGHT + y_offset : (y_offset >= MAP_HEIGHT ? y_offset - MAP_HEIGHT : y_offset);
          scale_neuron_at_position(x_coord, y_coord, sample, scale, total_components);
        }
      }
}

void free_allocated_memory()
{
  for (int i = 0; i < info.total_components; i++)
  {
    free(info.components[i].name);
    free(info.components[i].max_value_str);
    free(info.components[i].min_value_str);
  }
  free(info.components);

  for (int i = 0; i < info.total_dataset_samples; i++)
    free(info.samples[i].components);
  free(info.samples);

  for (int x = 0; x < MAP_WIDTH; x++)
  {
    for (int y = 0; y < MAP_HEIGHT; y++)
      free(map[x][y].weights);
    free(map[x]);
  }
  free(map);
}

unsigned long createRGBA(int r, int g, int b, int a)
{
  return ((r & 0xff) << 24) + ((g & 0xff) << 16) + ((b & 0xff) << 8) + (a & 0xff);
}

void clear_texture(RenderTexture2D *texture, Color color)
{
  BeginDrawing();
  BeginTextureMode(*texture);
  ClearBackground(color);
  EndTextureMode();
  EndDrawing();
}

void update_heightmap_3d(RenderTexture2D *render_texture, int component_index)
{
  static const Camera camera = {{18.0f, 18.0f, 18.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 45.0f, 0};
  static const Vector3 mapPosition = {-8.0f, 0.0f, -8.0f};
  Image image = LoadImageFromTexture(render_texture->texture);
  Mesh mesh = GenMeshHeightmap(image, (Vector3){16, 8, 16});
  UnloadImage(image);
  Model model = LoadModelFromMesh(mesh);
  model.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = render_texture->texture;

  BeginDrawing();

  BeginMode3D(camera);
  ClearBackground(GREEN);
  DrawModel(model, mapPosition, 1.0f, RED);
  EndMode3D();

  BeginTextureMode(*render_texture);

  for (int y = 0; y < MAP_HEIGHT; y++)
    for (int x = 0; x < MAP_WIDTH; x++)
      DrawPixel(x, y, GetColor(createRGBA((int)(map[x][y].weights[component_index] * 255), 0, 0, 255)));

  EndTextureMode();
  DrawTexturePro(render_texture->texture, (Rectangle){0, 0, (float)render_texture->texture.width, (float)-render_texture->texture.height}, (Rectangle){SCREEN_WIDTH - 210, 10, 200, 200}, (Vector2){0.0f, 0.0f}, 0, WHITE);

  EndDrawing();
}

void draw_text(RenderTexture2D *render_texture, char *text)
{
  BeginDrawing();
  BeginTextureMode(*render_texture);
  DrawText(text, 10, 10, 20, RAYWHITE);
  EndTextureMode();
  DrawTexturePro(render_texture->texture, (Rectangle){0, 0, (float)render_texture->texture.width, (float)-render_texture->texture.height}, (Rectangle){0, 0, MAP_LAYOUT_WIDTH, MAP_LAYOUT_HEIGHT}, (Vector2){0.0f, 0.0f}, 0, WHITE);
  EndDrawing();
}

void update_texture(RenderTexture2D *render_texture, int component_index, Neuron *neuron_at_mouse_position)
{
  BeginDrawing();
  BeginTextureMode(*render_texture);

  for (int y = 0; y < MAP_HEIGHT; y++)
    for (int x = 0; x < MAP_WIDTH; x++)
      DrawPixel(x, y, GetColor(createRGBA((int)(map[x][y].weights[component_index] * 255), 0, 0, 255)));

  DrawText(info.components[component_index].name, 10, 10, 20, RAYWHITE);

  for (int x = 0; x < MAP_WIDTH; x++)
    DrawLineEx((Vector2){x, MAP_HEIGHT - 12}, (Vector2){x, MAP_HEIGHT - 3}, 1.0f, GetColor(createRGBA((int)((x * 255) / MAP_WIDTH), 0, 0, 255)));

  if (neuron_at_mouse_position != NULL)
  {
    // Draw the green indicator according to the weight of the neuron being pointed by the mouse cursor
    int indicator_x = (int)(neuron_at_mouse_position->weights[component_index] * MAP_WIDTH);
    DrawLineEx((Vector2){indicator_x, MAP_HEIGHT - 12}, (Vector2){indicator_x, MAP_HEIGHT - 3}, 1.0f, GREEN);

    // Calculate and draw the value that corresponds to the neuron being pointed by the mouse cursor
    static char value_str[50];
    double value = ((info.components[component_index].max_value - info.components[component_index].min_value) * neuron_at_mouse_position->weights[component_index]) + info.components[component_index].min_value;
    snprintf(value_str, 50, "%f", value);
    DrawText(value_str, (MAP_WIDTH/2)-10, MAP_HEIGHT - 25, 1, RAYWHITE);
  }

  DrawText(info.components[component_index].min_value_str, 2, MAP_HEIGHT - 12, 1, RAYWHITE);
  DrawText(info.components[component_index].max_value_str, MAP_WIDTH - 30, MAP_HEIGHT - 12, 1, RAYWHITE);

  EndTextureMode();
  DrawTexturePro(render_texture->texture, (Rectangle){0, 0, (float)render_texture->texture.width, (float)-render_texture->texture.height}, (Rectangle){0, 0, MAP_LAYOUT_WIDTH, MAP_LAYOUT_HEIGHT}, (Vector2){0.0f, 0.0f}, 0, WHITE);
  EndDrawing();
}

void update_samples_texture(RenderTexture2D *render_texture)
{
  BeginDrawing();
  BeginTextureMode(*render_texture);
  ClearBackground(WHITE);

  for (int i = 0; i < info.total_dataset_samples; i++)
  {
    if (info.samples[i].value >= 8)
    {
      // Red pixels represents high quality wine samples
      DrawPixel(info.samples[i].bmu.x_coord, info.samples[i].bmu.y_coord, GetColor(createRGBA(255, 0, 0, 255)));
    }
    else if (info.samples[i].value >= 6)
    {
      // Green pixels represents average quality wine samples
      DrawPixel(info.samples[i].bmu.x_coord, info.samples[i].bmu.y_coord, GetColor(createRGBA(0, 255, 0, 255)));
    }
    else if (info.samples[i].value >= 0)
    {
      // Blue pixels represents poor quality wine samples
      DrawPixel(info.samples[i].bmu.x_coord, info.samples[i].bmu.y_coord, GetColor(createRGBA(0, 0, 255, 255)));
    }
  }

  EndTextureMode();
  DrawTexturePro(render_texture->texture, (Rectangle){0, 0, (float)render_texture->texture.width, (float)-render_texture->texture.height}, (Rectangle){0, 0, MAP_LAYOUT_WIDTH, MAP_LAYOUT_HEIGHT}, (Vector2){0.0f, 0.0f}, 0, WHITE);
  EndDrawing();
}

void update_text_texture(RenderTexture2D *texture, int selected_component_index, bool training_finished)
{
  BeginDrawing();
  BeginTextureMode(*texture);
  ClearBackground(BLACK);

  DrawText("Components:", 10, 10, 40, RAYWHITE);

  static char str_component_index[100];
  for (int i = 0; i < info.total_components - 1; i++)
  {
    sprintf(str_component_index, "%d %s", i, info.components[i].name);
    DrawText(str_component_index, 10, 70 + (44 * i), 32, LIME);
  }
  DrawText("Press the number key or", 10, 70 + (44 * info.total_components) + 20, 28, RAYWHITE);
  DrawText("use UP and DOWN keys to", 10, 70 + (44 * info.total_components) + 60, 28, RAYWHITE);
  DrawText("select a component.", 10, 70 + (44 * info.total_components) + 100, 28, RAYWHITE);

  if (!training_finished)
  {
    DrawText("Press ENTER key to stop", 10, 70 + (44 * info.total_components) + 180, 28, YELLOW);
    DrawText("training and run inference.", 10, 70 + (44 * info.total_components) + 210, 28, YELLOW);
  }
  else
  {
    DrawText("Press RIGHT SHIFT key to", 10, 70 + (44 * info.total_components) + 180, 28, YELLOW);
    DrawText("show inferenced results.", 10, 70 + (44 * info.total_components) + 210, 28, YELLOW);
  }

  DrawText("Use LEFT and RIGHT keys", 10, 70 + (44 * info.total_components) + 290, 28, RAYWHITE);
  DrawText("to change the marker color. ", 10, 70 + (44 * info.total_components) + 320, 28, RAYWHITE);
  DrawText("Press SPACE bar to clean", 10, 70 + (44 * info.total_components) + 350, 28, RAYWHITE);
  DrawText("marker marks.", 10, 70 + (44 * info.total_components) + 380, 28, RAYWHITE);

  EndTextureMode();
  DrawTexturePro(texture->texture, (Rectangle){0, 0, 400, -1200}, (Rectangle){MAP_LAYOUT_WIDTH, 0, 300, 900}, (Vector2){0.0f, 0.0f}, 0, WHITE);

  EndDrawing();
}

void update_colorpicker_texture(RenderTexture2D *paint_render, int color_selected, Color *colors, Rectangle *color_rectangles)
{
  BeginDrawing();
  DrawTextureRec(paint_render->texture, (Rectangle){0, 0, (float)paint_render->texture.width, (float)-paint_render->texture.height}, (Vector2){0, 0}, WHITE);
  // Draw top panel
  DrawRectangle(MAP_LAYOUT_WIDTH, MAP_LAYOUT_HEIGHT - 90, SCREEN_WIDTH, 90, RAYWHITE);

  // Draw color selection rectangles
  for (int i = 0; i < MAX_COLORS_COUNT; i++)
    DrawRectangleRec(color_rectangles[i], colors[i]);

  DrawRectangleLinesEx((Rectangle){color_rectangles[color_selected].x - 2, color_rectangles[color_selected].y - 2, color_rectangles[color_selected].width + 4, color_rectangles[color_selected].height + 4}, 2, BLACK);
  EndDrawing();
}

void process_key_pressed(int *selected_component_index, Neuron *neuron_at_mouse_position, bool *training_finished, int *color_selected, bool *show_3d_surface_plot, bool *show_samples_in_map, RenderTexture2D *render_texture, RenderTexture2D *texture, RenderTexture2D *marker_texture)
{
  int key_pressed = GetKeyPressed();
  bool text_need_update = false;
  bool color_changed = false;
  if ((key_pressed == 265 /*UP*/) && (*selected_component_index > 0))
  {
    (*selected_component_index)--;
    text_need_update = true;
  }
  else if ((key_pressed == 264 /*DOWN*/) && (*selected_component_index < info.total_components - 2))
  {
    (*selected_component_index)++;
    text_need_update = true;
  }
  else if ((key_pressed >= 48) && (key_pressed <= 57)) // Number key
  {
    // Change the component to render
    *selected_component_index = key_pressed - 48;
    text_need_update = true;
  }
  else if (key_pressed == 257) // ENTER key
  {
    // Finish training
    *training_finished = true;
    text_need_update = true;
  }
  else if (key_pressed == 32) // SPACE key
  {
    // Clean marker marks
    clear_texture(marker_texture, BLANK);
  }
  else if (key_pressed == 86) // V key
  {
    // Enable/disable 3D view
    *show_3d_surface_plot = !*show_3d_surface_plot;
    text_need_update = true;
  }
  else if (key_pressed == 262) // RIGHT key
  {
    // Select next color
    (*color_selected)++;
    color_changed = true;
  }
  else if (key_pressed == 263) // LEFT key
  {
    // Select previous color
    (*color_selected)--;
    color_changed = true;
  }
  else if ((key_pressed == 344) && *training_finished) // Right Shift
  {
    // Show inferenced results
    *show_samples_in_map = true;
    update_samples_texture(render_texture);
  }

  if (color_changed)
  {
    if (*color_selected >= MAX_COLORS_COUNT)
      *color_selected = MAX_COLORS_COUNT - 1;
    else if (*color_selected < 0)
      *color_selected = 0;
  }

  if (text_need_update)
  {
    *show_samples_in_map = false;

    if (*show_3d_surface_plot)
    {
      update_heightmap_3d(render_texture, *selected_component_index);
    }
    else
    {
      update_texture(render_texture, *selected_component_index, neuron_at_mouse_position);
      update_text_texture(texture, *selected_component_index, *training_finished);
    }
  }
}

void process_mouse_events(Vector2 *prev_mouse_position, Vector2 *prev_mouse_click_position, bool *mouse_button_is_pressed, Neuron **neuron_at_mouse_position, bool show_samples_in_map, bool training_finished, int selected_component_index, int color_selected, Color *colors, RenderTexture2D *paint_render, RenderTexture2D *render_texture)
{
  Vector2 current_mouse_position = GetMousePosition();
  if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) || (GetGestureDetected() == GESTURE_DRAG))
  {
    BeginTextureMode(*paint_render);
    if (*mouse_button_is_pressed)
      DrawLineEx(*prev_mouse_click_position, current_mouse_position, 5.0f, colors[color_selected]);
    *prev_mouse_click_position = current_mouse_position;
    *mouse_button_is_pressed = true;
    EndTextureMode();
  }
  else
  {
    *mouse_button_is_pressed = false;
  }

  bool indicator_updated = false;
  if (training_finished)
  {
    bool mouse_is_out_of_map_layout = (current_mouse_position.x > MAP_LAYOUT_WIDTH) || (current_mouse_position.y > MAP_LAYOUT_HEIGHT);
    if (((current_mouse_position.x != prev_mouse_position->x) || (current_mouse_position.y != prev_mouse_position->y)) && !mouse_is_out_of_map_layout)
    {
      int x = min((current_mouse_position.x * MAP_WIDTH) / MAP_LAYOUT_WIDTH, MAP_WIDTH - 1);
      int y = min((current_mouse_position.y * MAP_HEIGHT) / MAP_LAYOUT_HEIGHT, MAP_HEIGHT - 1);
      *neuron_at_mouse_position = &map[x][y];
      *prev_mouse_position = current_mouse_position;
      indicator_updated = true;
    }
    else if (mouse_is_out_of_map_layout)
    {
      *neuron_at_mouse_position = NULL;
    }
  }

  if (!show_samples_in_map && (*mouse_button_is_pressed || indicator_updated))
    update_texture(render_texture, selected_component_index, *neuron_at_mouse_position);
}

void initialize_color_rectangles(Rectangle *color_rectangles)
{
  for (int i = 0; i < MAX_COLORS_COUNT; i++)
  {
    color_rectangles[i].x = MAP_LAYOUT_WIDTH + 10 + 30.0f * (i % 9) + 2 * (i % 9);
    color_rectangles[i].y = i < 9 ? MAP_LAYOUT_HEIGHT - 80 : MAP_LAYOUT_HEIGHT - 40;
    color_rectangles[i].width = 30;
    color_rectangles[i].height = 30;
  }
}

int main(int argc, char *argv[])
{
  char title[100] = "SOM";
  InitWindow(SCREEN_WIDTH, min(SCREEN_HEIGHT, MAP_LAYOUT_HEIGHT), title);
  RenderTexture2D render_texture = LoadRenderTexture(MAP_WIDTH, MAP_HEIGHT);
  RenderTexture2D text_texture = LoadRenderTexture(400, 1200);
  RenderTexture2D paint_render = LoadRenderTexture(MAP_LAYOUT_WIDTH, MAP_LAYOUT_HEIGHT);

  // Load and initialize info and samples from the dataset
  load_dataset(dataset_csv_file);

  BMU bmu;
  Sample *sample;
  double learning_rule = INITIAL_LEARNING_RULE;
  double radius = INITIAL_RADIUS;
  int key_pressed, selected_component_index = 0;
  bool training_finished = false;
  bool application_finished = false;
  bool show_3d_surface_plot = false;
  bool show_samples_in_map = false;

  Vector2 prev_mouse_position, prev_mouse_click_position;
  Neuron *neuron_at_mouse_position = NULL;
  bool mouse_button_is_pressed = false;
  Color colors[MAX_COLORS_COUNT] = {RAYWHITE, YELLOW, GOLD, ORANGE, PINK, RED, MAROON, GREEN, LIME, DARKGREEN, SKYBLUE, BLUE, DARKBLUE, PURPLE, VIOLET, DARKPURPLE, BEIGE, BROWN};
  int color_selected = 0;

  // Initialize color palette rectangles
  Rectangle color_rectangles[MAX_COLORS_COUNT] = {0};
  initialize_color_rectangles(color_rectangles);

  // Random seed
  srand(time(NULL));

  // Initialize the Neural Network (Self-Organizing Map)
  initialize_som_map();

  update_text_texture(&text_texture, selected_component_index, training_finished);

  epoch = 0;
  while ((epoch < TOTAL_EPOCHS) && !training_finished && !application_finished)
  {
    radius = max(1.0L, (epoch == 0) ? INITIAL_RADIUS : (radius - (radius / 3.0L)));
    learning_rule = max(0.015L, INITIAL_LEARNING_RULE * exp(-10.0L * (epoch * epoch) / (TOTAL_EPOCHS * TOTAL_EPOCHS)));
    iterations_per_epoch = (epoch == 0) ? INITIAL_TRAINING_ITERATIONS_PER_EPOCH : (iterations_per_epoch * 2);
    epoch++;

    iteration = 0;
    while ((iteration < iterations_per_epoch) && !training_finished && !application_finished)
    {
      sample = pick_random_sample();
      search_bmu(sample, &bmu, info.total_components); // search for the Best Match Unit
      scale_neighbors(&bmu, sample, radius, learning_rule, info.total_components);

      iteration++;
      if (show_3d_surface_plot)
        update_heightmap_3d(&render_texture, selected_component_index);
      else
        update_texture(&render_texture, selected_component_index, neuron_at_mouse_position);

      process_key_pressed(&selected_component_index, neuron_at_mouse_position, &training_finished, &color_selected, &show_3d_surface_plot, &show_samples_in_map, &render_texture, &text_texture, &paint_render);
      process_mouse_events(&prev_mouse_position, &prev_mouse_click_position, &mouse_button_is_pressed, &neuron_at_mouse_position, show_samples_in_map, training_finished, selected_component_index, color_selected, colors, &paint_render, &render_texture);

      if (!show_3d_surface_plot)
        update_colorpicker_texture(&paint_render, color_selected, colors, color_rectangles);

      sprintf(title, "EPOCH %d/%d | ITERATION: %d/%d | RADIUS: %.2f | LEARNING RULE: %.4f", epoch, TOTAL_EPOCHS, iteration, iterations_per_epoch, radius, learning_rule);
      SetWindowTitle(title);

      if (WindowShouldClose())
        application_finished = true;
    }
  }

  training_finished = true;
  show_samples_in_map = true;

  if (!application_finished)
  {
    clear_texture(&render_texture, BLACK);
    draw_text(&render_texture, "Inferencing samples...");
    SetWindowTitle("Please wait while running inference...");

    // Calculate inference for each sample of the dataset
    for (int i = 0; !WindowShouldClose() && (i < info.total_dataset_samples); i++)
    {
      search_bmu(&(info.samples[i]), &bmu, info.total_components);
      info.samples[i].bmu.x_coord = bmu.x_coord;
      info.samples[i].bmu.y_coord = bmu.y_coord;
    }

    // Render dataset samples in map
    update_samples_texture(&render_texture);

    SetWindowTitle("Inferenced results");
    SetTargetFPS(30);
    while (!WindowShouldClose() && !application_finished)
    {
      process_key_pressed(&selected_component_index, neuron_at_mouse_position, &training_finished, &color_selected, &show_3d_surface_plot, &show_samples_in_map, &render_texture, &text_texture, &paint_render);
      process_mouse_events(&prev_mouse_position, &prev_mouse_click_position, &mouse_button_is_pressed, &neuron_at_mouse_position, show_samples_in_map, training_finished, selected_component_index, color_selected, colors, &paint_render, &render_texture);
      update_colorpicker_texture(&paint_render, color_selected, colors, color_rectangles);

      if (WindowShouldClose())
        application_finished = true;
    }
  }

  free_allocated_memory();
  UnloadRenderTexture(render_texture);
  UnloadRenderTexture(paint_render);
  CloseWindow();

  return 0;
}
