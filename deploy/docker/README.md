# Pre-requisites

* To build or run docker images, we need to have docker installed in our system and running.

Important note on Docker:

If you are using the ubuntu App, then some additional configuration is needed before using docker client from the ubuntu App. See the instructions here:
https://nickjanetakis.com/blog/setting-up-docker-for-windows-and-wsl-to-work-flawlessly

** Getting Started **

## Building Docker images

- To build docker, run the invoke task from base environment

    ```
    (base):~/code_templates$ inv dev.build-docker
    ```

This builds the image to run productions scripts present in the current code-archive and the image created will be `ct-reg-py` for regression archive, `ct-class-py` for classification and so on

- To find the list of docker images, run

    ```
    (base):~/code_templates$ docker images -a 
    ```

## Running the built docker production image

```
(base):~/code_templates$ docker run -it <image-name> 

```
Similarly, the other production script commands can be run on the above image as:

```
(base):~/code_templates$ docker run -it <image-name> job run --job-id <task name>

```

## Mount Data to docker

If you want to pass any data to docker container, then we can mount the local directories on to docker container as below with `-v` option during docker run
```
(base):~/code_templates$ docker run -it -v <local-dir>:<container-dir> <image-name> 

```
Example:

```
(base):~/code_templates$ docker run -it -v D:\regression-py\data\:/home/app/app/data  ct-reg-py job run --job-id data-cleaning

```
In the above example, docker performs `data-cleaning` on the data I mounted on to the container.

Here is a Docker cheat sheet that can be useful: http://dockerlabs.collabnix.com/docker/cheatsheet/


