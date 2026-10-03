FROM gcc:14
WORKDIR /app
COPY . .
RUN g++ -std=c++17 -O2 main.cpp -o app -pthread
ENV PORT=10000
EXPOSE 10000
CMD ["./app"]
