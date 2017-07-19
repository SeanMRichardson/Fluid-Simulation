#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

#include <random>

#define MAX_EPSILON 0.10f
#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 512;
const unsigned int window_height = 512;

const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

const char *volumeFilename = "Bucky.raw";
const cudaExtent volumeSize = make_cudaExtent(32, 32, 32);

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

//GLuint nVbo;
//struct cudaGraphicsResource *cuda_norm_resource;
//void *d_norm_buffer = NULL;

GLuint indexBuffer;
GLuint shaderProg;
char *vertShaderPath = 0, *fragShaderPath = 0;

float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

float *g_hptr = NULL;
float4 *pptr = 0;
//float4 *nptr = 0;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

extern "C" void initCuda(const unsigned char *h_volume, cudaExtent volumeSize);
extern "C" void calculate_position_kernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time);
//extern "C" void calculate_normal_kernel(float4 *pos, float4 *norms, unsigned int mesh_width, unsigned int mesh_height);

void loadVolumeData(char *exec_path);

void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);

		sdkResetTimer(&timer);
	}

	char fps[256];
	sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
	glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
//void createNVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

void createVBO(GLuint *vbo, int size);
void deleteVBO(GLuint *vbo);
void createMeshIndexBuffer(GLuint *id, int w, int h);
void createMeshPositionVBO(GLuint *id, int w, int h);
GLuint loadGLSLProgram(const char *vertFileName, const char *fragFileName);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda();
void runAutoTest(int devID, char **argv, char *ref_file);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

const char *sSDKsample = "simpleGL (VBO)";

bool checkHW(char *name, const char *gpuType, int dev)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	strcpy(name, deviceProp.name);

	if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
	{
		return true;
	}
	else
	{
		return false;
	}
}

int findGraphicsGPU(char *name)
{
	int nGraphicsGPU = 0;
	int deviceCount = 0;
	bool bFoundGraphics = false;
	char firstGraphicsName[256], temp[256];

	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("> FAILED %s sample finished, exiting...\n", sSDKsample);
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		printf("> There are no device(s) supporting CUDA\n");
		return false;
	}
	else
	{
		printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
	}

	for (int dev = 0; dev < deviceCount; ++dev)
	{
		bool bGraphics = !checkHW(temp, (const char *)"Tesla", dev);
		printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

		if (bGraphics)
		{
			if (!bFoundGraphics)
			{
				strcpy(firstGraphicsName, temp);
			}

			nGraphicsGPU++;
		}
	}

	if (nGraphicsGPU)
	{
		strcpy(name, firstGraphicsName);
	}
	else
	{
		strcpy(name, "this hardware");
	}

	return nGraphicsGPU;
}

// Load raw data from disk
unsigned char *loadRawFile(const char *filename, size_t size)
{
	FILE *fp = fopen(filename, "rb");

	if (!fp)
	{
		fprintf(stderr, "Error opening file '%s'\n", filename);
		return 0;
	}

	unsigned char *data = (unsigned char *)malloc(size);
	size_t read = fread(data, 1, size, fp);
	fclose(fp);

	printf("Read '%s', %lu bytes\n", filename, (unsigned long)read);

	return data;
}

void loadVolumeData(char *exec_path)
{
	// load volume data
	const char *path = sdkFindFilePath(volumeFilename, exec_path);

	if (path == NULL)
	{
		fprintf(stderr, "Error unable to find 3D Volume file: '%s'\n", volumeFilename);
		exit(EXIT_FAILURE);
	}

	size_t size = volumeSize.width*volumeSize.height*volumeSize.depth;
	unsigned char *h_volume = loadRawFile(path, size);

	initCuda(h_volume, volumeSize);
	sdkCreateTimer(&timer);

	free(h_volume);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	char *ref_file = NULL;

	pArgc = &argc;
	pArgv = argv;

#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif

	printf("%s starting...\n", sSDKsample);

	if (argc > 1)
	{
		if (checkCmdLineFlag(argc, (const char **)argv, "file"))
		{
			// In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
			getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
		}
	}

	printf("\n");

	runTest(argc, argv, ref_file);

	printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
	exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda GL Interop (VBO)");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	vertShaderPath = sdkFindFilePath("ocean.vert", argv[0]);
	fragShaderPath = sdkFindFilePath("ocean.frag", argv[0]);

	if (vertShaderPath == NULL || fragShaderPath == NULL)
	{
		fprintf(stderr, "Error unable to find GLSL vertex and fragment shaders!\n");
		exit(EXIT_FAILURE);
	}

	// initialize necessary OpenGL extensions
	if (!isGLVersionSupported(2, 0))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glEnable(GL_DEPTH_TEST);

	// load shader
	shaderProg = loadGLSLProgram(vertShaderPath, fragShaderPath);

	//glBindAttribLocation(shaderProg, 0, "normal");

	SDK_CHECK_ERROR_GL();

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, int size)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo)
{
	glDeleteBuffers(1, vbo);
	*vbo = 0;
}

// create index buffer for rendering quad mesh
void createMeshIndexBuffer(GLuint *id, int w, int h)
{
	int size = ((w * 2) + 2)*(h - 1) * sizeof(GLuint);

	// create index buffer
	glGenBuffers(1, id);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *id);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

	// fill with indices for rendering mesh as triangle strips
	GLuint *indices = (GLuint *)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);

	if (!indices)
	{
		return;
	}

	for (int y = 0; y<h - 1; y++)
	{
		for (int x = 0; x<w; x++)
		{
			*indices++ = y*w + x;
			*indices++ = (y + 1)*w + x;
		}

		// start new strip with degenerate triangle
		*indices++ = (y + 1)*w + (w - 1);
		*indices++ = (y + 1)*w;
	}

	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

// create fixed vertex buffer to store mesh vertices
void createMeshPositionVBO(GLuint *id, int w, int h)
{
	createVBO(id, w*h * 4 * sizeof(float));

	glBindBuffer(GL_ARRAY_BUFFER, *id);
	float *pos = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	if (!pos)
	{
		return;
	}

	for (int y = 0; y<h; y++)
	{
		for (int x = 0; x<w; x++)
		{
			float u = x / (float)(w - 1);
			float v = y / (float)(h - 1);
			*pos++ = u*2.0f - 1.0f;
			*pos++ = 0.0f;
			*pos++ = v*2.0f - 1.0f;
			*pos++ = 1.0f;
		}
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Attach shader to a program
int attachShader(GLuint prg, GLenum type, const char *name)
{
	GLuint shader;
	FILE *fp;
	int size, compiled;
	char *src;

	fp = fopen(name, "rb");

	if (!fp)
	{
		return 0;
	}

	fseek(fp, 0, SEEK_END);
	size = ftell(fp);
	src = (char *)malloc(size);

	fseek(fp, 0, SEEK_SET);
	fread(src, sizeof(char), size, fp);
	fclose(fp);

	shader = glCreateShader(type);
	glShaderSource(shader, 1, (const char **)&src, (const GLint *)&size);
	glCompileShader(shader);
	glGetShaderiv(shader, GL_COMPILE_STATUS, (GLint *)&compiled);

	if (!compiled)
	{
		char log[2048];
		int len;

		glGetShaderInfoLog(shader, 2048, (GLsizei *)&len, log);
		printf("Info log: %s\n", log);
		glDeleteShader(shader);
		return 0;
	}

	free(src);

	glAttachShader(prg, shader);
	glDeleteShader(shader);

	return 1;
}

// Create shader program from vertex shader and fragment shader files
GLuint loadGLSLProgram(const char *vertFileName, const char *fragFileName)
{
	GLint linked;
	GLuint program;

	program = glCreateProgram();

	if (!attachShader(program, GL_VERTEX_SHADER, vertFileName))
	{
		glDeleteProgram(program);
		fprintf(stderr, "Couldn't attach vertex shader from file %s\n", vertFileName);
		return 0;
	}

	if (!attachShader(program, GL_FRAGMENT_SHADER, fragFileName))
	{
		glDeleteProgram(program);
		fprintf(stderr, "Couldn't attach fragment shader from file %s\n", fragFileName);
		return 0;
	}

	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &linked);

	if (!linked)
	{
		glDeleteProgram(program);
		char temp[256];
		glGetProgramInfoLog(program, 256, 0, temp);
		fprintf(stderr, "Failed to link program: %s\n", temp);
		return 0;
	}

	return program;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file)
{
	// Create the CUTIL timer
	sdkCreateTimer(&timer);

	// command line mode only
	if (ref_file != NULL)
	{
		// This will pick the best possible CUDA capable device
		int devID = findCudaDevice(argc, (const char **)argv);

		// create VBO
		checkCudaErrors(cudaMalloc((void **)&d_vbo_buffer, mesh_width*mesh_height * 4 * sizeof(float)));

		loadVolumeData(argv[0]);

		// run the cuda part
		runAutoTest(devID, argv, ref_file);

		// check result of Cuda step
		checkResultCuda(argc, argv, vbo);

		cudaFree(d_vbo_buffer);
		d_vbo_buffer = NULL;
	}
	else
	{
		// First initialize OpenGL context, so we can properly set the GL for CUDA.
		// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
		if (false == initGL(&argc, argv))
		{
			return false;
		}

		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
		if (checkCmdLineFlag(argc, (const char **)argv, "device"))
		{
			if (gpuGLDeviceInit(argc, (const char **)argv) == -1)
			{
				return false;
			}
		}
		else
		{
			cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
		}

		// register callbacks
		glutDisplayFunc(display);
		glutKeyboardFunc(keyboard);
		glutMouseFunc(mouse);
		glutMotionFunc(motion);
#if defined (__APPLE__) || defined(MACOSX)
		atexit(cleanup);
#else
		glutCloseFunc(cleanup);
#endif
		loadVolumeData(argv[0]);


		// create vertex and index buffer for mesh
		// create VBO
		createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
		//createNVBO(&nVbo, &cuda_norm_resource, cudaGraphicsMapFlagsWriteDiscard);

		//createMeshPositionVBO(&vbo, mesh_width, mesh_height);
		createMeshIndexBuffer(&indexBuffer, mesh_width, mesh_height);

		// run the cuda part
		runCuda();

		// start rendering mainloop
		glutMainLoop();
	}

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
	// map OpenGL buffer object for writing from CUDA
	
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&pptr, &num_bytes, cuda_vbo_resource));

	calculate_position_kernel(pptr, mesh_width, mesh_height, g_fAnim);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

	/*checkCudaErrors(cudaGraphicsMapResources(1, &cuda_norm_resource, 0));

	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&nptr, &num_bytes, cuda_norm_resource));

	calculate_normal_kernel(pptr, nptr, mesh_width, mesh_height);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_norm_resource, 0));*/
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif

void sdkDumpBin2(void *data, unsigned int bytes, const char *filename)
{
	printf("sdkDumpBin: <%s>\n", filename);
	FILE *fp;
	FOPEN(fp, filename, "wb");
	fwrite(data, bytes, 1, fp);
	fflush(fp);
	fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int devID, char **argv, char *ref_file)
{
	char *reference_file = NULL;
	void *imageData = malloc(mesh_width*mesh_height * sizeof(float));

	// execute the kernel
	calculate_position_kernel((float4 *)d_vbo_buffer, mesh_width, mesh_height, g_fAnim);

	cudaDeviceSynchronize();
	getLastCudaError("launch_kernel failed");

	checkCudaErrors(cudaMemcpy(imageData, d_vbo_buffer, mesh_width*mesh_height * sizeof(float), cudaMemcpyDeviceToHost));


	sdkDumpBin2(imageData, mesh_width*mesh_height * sizeof(float), "simpleGL.bin");
	reference_file = sdkFindFilePath(ref_file, argv[0]);

	if (reference_file &&
		!sdkCompareBin2BinFloat("simpleGL.bin", reference_file,
			mesh_width*mesh_height * sizeof(float),
			MAX_EPSILON_ERROR, THRESHOLD, pArgv[0]))
	{
		g_TotalErrors++;
	}
	checkCudaErrors(cudaFree(g_hptr));
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createNVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	sdkStartTimer(&timer);

	// run CUDA kernel to generate vertex positions
	runCuda();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	//glClientActiveTexture(GL_TEXTURE0);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	//glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	/*glBindBuffer(GL_ARRAY_BUFFER, nVbo);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);*/

	//glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glUseProgram(shaderProg);

	// Set default uniform variables parameters for the vertex shader
	GLuint uniHeightScale, uniSize;

	uniHeightScale = glGetUniformLocation(shaderProg, "heightScale");
	glUniform1f(uniHeightScale, 0.5f);

	uniSize = glGetUniformLocation(shaderProg, "size");
	glUniform2f(uniSize, (float)mesh_width, (float)mesh_height);

	GLuint uniLightDir;

	uniLightDir = glGetUniformLocation(shaderProg, "lightDir");
	glUniform3f(uniLightDir, 0.0f, 1.0f, 0.0f);

	//
	/*glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDrawElements(GL_TRIANGLE_STRIP, ((mesh_width * 2) + 2)*(mesh_height - 1), GL_UNSIGNED_INT, 0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);*/

	//
	//glClientActiveTexture(GL_TEXTURE0);
	//glDisableClientState(GL_TEXTURE_COORD_ARRAY);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);

	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
	glDisableClientState(GL_VERTEX_ARRAY);

	glUseProgram(0);

	glutSwapBuffers();

	g_fAnim += 0.01f;

	sdkStopTimer(&timer);
	computeFPS();
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}

void cleanup()
{
	sdkDeleteTimer(&timer);

	if (vbo)
	{
		deleteVBO(&vbo, cuda_vbo_resource);
	}

	/*if (nVbo)
	{
		deleteVBO(&nVbo, cuda_norm_resource);
	}*/
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (27):
#if defined(__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char **argv, const GLuint &vbo)
{
	if (!d_vbo_buffer)
	{
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

		// map buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float *data = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

		// check result
		if (checkCmdLineFlag(argc, (const char **)argv, "regression"))
		{
			// write file for regression test
			sdkWriteFile<float>("./data/regression.dat",
				data, mesh_width * mesh_height * 3, 0.0, false);
		}

		// unmap GL buffer object
		if (!glUnmapBuffer(GL_ARRAY_BUFFER))
		{
			fprintf(stderr, "Unmap buffer failed.\n");
			fflush(stderr);
		}

		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
			cudaGraphicsMapFlagsWriteDiscard));

		SDK_CHECK_ERROR_GL();
	}
}

