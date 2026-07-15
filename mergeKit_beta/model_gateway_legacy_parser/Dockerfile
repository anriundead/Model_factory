FROM maven:3.9-eclipse-temurin-21 AS build

WORKDIR /src
COPY mergeKit_beta/model_gateway_legacy_parser/pom.xml ./
RUN mvn -B -q dependency:go-offline
COPY mergeKit_beta/model_gateway_legacy_parser/src ./src
RUN mvn -B -q test && mvn -B -q package -DskipTests && mvn -B -q dependency:copy-dependencies -DincludeScope=runtime -DoutputDirectory=target/dependency

FROM eclipse-temurin:17-jre

RUN useradd --uid 65532 --create-home --shell /usr/sbin/nologin parser
COPY --from=build /src/target/legacy-office-parser-1.0.0.jar /opt/legacy-parser/legacy-office-parser.jar
COPY --from=build /src/target/dependency /opt/legacy-parser/dependency

USER 65532:65532
WORKDIR /opt/legacy-parser
CMD ["java", "-Xms32m", "-Xmx384m", "-jar", "/opt/legacy-parser/legacy-office-parser.jar"]
