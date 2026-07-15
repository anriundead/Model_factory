package org.mergenetic.legacyparser;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import org.apache.poi.hslf.usermodel.HSLFSlide;
import org.apache.poi.hslf.usermodel.HSLFSlideShow;
import org.apache.poi.hslf.usermodel.HSLFTextShape;
import org.apache.poi.hwpf.HWPFDocument;
import org.apache.poi.hwpf.usermodel.Paragraph;
import org.apache.poi.hwpf.usermodel.Range;
import org.apache.poi.sl.usermodel.Shape;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public final class LegacyOfficeParser {
    private static final class ParseError extends Exception {
        ParseError(String message) { super(message); }
    }

    static record Section(String kind, int value, String text) {}

    private record Settings(String token, Path workRoot, int timeoutSeconds, long maxBytes) {}

    private LegacyOfficeParser() {}

    public static void main(String[] args) throws Exception {
        if (args.length == 1 && "--healthcheck".equals(args[0])) {
            healthcheck();
            return;
        }
        if (args.length == 3 && "--parse".equals(args[0])) {
            System.out.print(toJson(parseDocument(Path.of(args[2]), args[1])));
            return;
        }
        startServer();
    }

    private static void startServer() throws IOException {
        Path workRoot = Path.of(System.getenv().getOrDefault("MERGEKIT_LEGACY_PARSER_WORK_ROOT", "/work"));
        Files.createDirectories(workRoot);
        Settings settings = new Settings(
            System.getenv().getOrDefault("MERGEKIT_LEGACY_PARSER_TOKEN", ""),
            workRoot,
            positiveEnv("MERGEKIT_LEGACY_PARSER_TIMEOUT_SECONDS", 45),
            positiveEnv("MERGEKIT_LEGACY_PARSER_MAX_MIB", 50) * 1024L * 1024L
        );
        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", positiveEnv("PORT", 8090)), 0);
        server.createContext("/healthz", new HealthHandler());
        server.createContext("/parse", new ParseHandler(settings));
        server.setExecutor(Executors.newFixedThreadPool(1));
        server.start();
    }

    private static int positiveEnv(String name, int fallback) {
        try { return Math.max(1, Integer.parseInt(System.getenv().getOrDefault(name, String.valueOf(fallback)))); }
        catch (NumberFormatException ignored) { return fallback; }
    }

    static String locatorKindFor(String sourceFormat) throws ParseError {
        if (sourceFormat == null) throw new ParseError("unsupported_legacy_format");
        return switch (sourceFormat) {
            case "doc" -> "paragraph";
            case "ppt" -> "slide";
            default -> throw new ParseError("unsupported_legacy_format");
        };
    }

    static List<Section> parseDocument(Path source, String sourceFormat) throws Exception {
        return switch (sourceFormat) {
            case "doc" -> parseDoc(source);
            case "ppt" -> parsePpt(source);
            default -> throw new ParseError("unsupported_legacy_format");
        };
    }

    private static List<Section> parseDoc(Path source) throws Exception {
        List<Section> sections = new ArrayList<>();
        try (InputStream input = Files.newInputStream(source); HWPFDocument document = new HWPFDocument(input)) {
            Range range = document.getRange();
            for (int index = 0; index < range.numParagraphs(); index++) {
                Paragraph paragraph = range.getParagraph(index);
                String text = clean(paragraph.text());
                if (!text.isEmpty()) sections.add(new Section("paragraph", index + 1, text));
            }
        }
        if (sections.isEmpty()) throw new ParseError("no_text_layer");
        return sections;
    }

    private static List<Section> parsePpt(Path source) throws Exception {
        List<Section> sections = new ArrayList<>();
        try (InputStream input = Files.newInputStream(source); HSLFSlideShow presentation = new HSLFSlideShow(input)) {
            List<HSLFSlide> slides = presentation.getSlides();
            for (int index = 0; index < slides.size(); index++) {
                StringBuilder text = new StringBuilder();
                for (Shape<?, ?> shape : slides.get(index).getShapes()) {
                    if (shape instanceof HSLFTextShape textShape) append(text, textShape.getText());
                }
                String cleaned = clean(text.toString());
                if (!cleaned.isEmpty()) sections.add(new Section("slide", index + 1, cleaned));
            }
        }
        if (sections.isEmpty()) throw new ParseError("no_text_layer");
        return sections;
    }

    private static void append(StringBuilder output, String value) {
        if (value == null || value.isBlank()) return;
        if (!output.isEmpty()) output.append('\n');
        output.append(value);
    }

    private static String clean(String value) {
        return value == null ? "" : value.replace('\u0000', ' ').replace('\u0007', ' ').trim();
    }

    private static String toJson(List<Section> sections) {
        StringBuilder json = new StringBuilder("{\"sections\":[");
        for (int index = 0; index < sections.size(); index++) {
            if (index > 0) json.append(',');
            Section section = sections.get(index);
            json.append("{\"kind\":\"").append(escape(section.kind())).append("\",\"value\":")
                .append(section.value()).append(",\"text\":\"").append(escape(section.text())).append("\"}");
        }
        return json.append("]}").toString();
    }

    private static String escape(String value) {
        StringBuilder escaped = new StringBuilder();
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            switch (character) {
                case '\\' -> escaped.append("\\\\");
                case '"' -> escaped.append("\\\"");
                case '\n' -> escaped.append("\\n");
                case '\r' -> escaped.append("\\r");
                case '\t' -> escaped.append("\\t");
                default -> {
                    if (character < 0x20) escaped.append(String.format("\\u%04x", (int) character));
                    else escaped.append(character);
                }
            }
        }
        return escaped.toString();
    }

    private static final class HealthHandler implements HttpHandler {
        @Override public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equals(exchange.getRequestMethod())) { error(exchange, 404, "not_found"); return; }
            respond(exchange, 200, "text/plain", "ok".getBytes(StandardCharsets.UTF_8));
        }
    }

    private static final class ParseHandler implements HttpHandler {
        private final Settings settings;
        ParseHandler(Settings settings) { this.settings = settings; }

        @Override public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equals(exchange.getRequestMethod())) { error(exchange, 404, "not_found"); return; }
            if (!authorized(settings.token(), exchange.getRequestHeaders().getFirst("X-Worker-Token"))) { error(exchange, 401, "unauthorized"); return; }
            String sourceFormat = exchange.getRequestHeaders().getFirst("X-Source-Format");
            try { locatorKindFor(sourceFormat); }
            catch (ParseError exception) { error(exchange, 400, exception.getMessage()); return; }
            long contentLength;
            try { contentLength = Long.parseLong(exchange.getRequestHeaders().getFirst("Content-Length")); }
            catch (RuntimeException exception) { error(exchange, 400, "invalid_request"); return; }
            if (contentLength < 1 || contentLength > settings.maxBytes()) { error(exchange, 413, "source_too_large"); return; }
            try {
                Path job = Files.createTempDirectory(settings.workRoot(), "parse-");
                try {
                    Path source = job.resolve("source." + sourceFormat);
                    if (copyExact(exchange.getRequestBody(), source, contentLength) != contentLength) { error(exchange, 400, "invalid_request"); return; }
                    byte[] payload = invokeChild(sourceFormat, source, job, settings);
                    respond(exchange, 200, "application/json", payload);
                } finally { deleteTree(job); }
            } catch (ParseError exception) { error(exchange, 422, exception.getMessage()); }
            catch (InterruptedException exception) { Thread.currentThread().interrupt(); error(exchange, 503, "legacy_parser_unavailable"); }
            catch (Exception exception) { error(exchange, 422, "legacy_parse_failed"); }
        }
    }

    private static long copyExact(InputStream input, Path output, long expected) throws IOException {
        long total = 0;
        byte[] buffer = new byte[8192];
        try (OutputStream stream = Files.newOutputStream(output)) {
            while (total < expected) {
                int read = input.read(buffer, 0, (int) Math.min(buffer.length, expected - total));
                if (read < 0) break;
                stream.write(buffer, 0, read);
                total += read;
            }
        }
        return total;
    }

    private static byte[] invokeChild(String sourceFormat, Path source, Path job, Settings settings) throws Exception {
        Path output = job.resolve("response.json");
        URI location = LegacyOfficeParser.class.getProtectionDomain().getCodeSource().getLocation().toURI();
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        Process process = new ProcessBuilder(java, "-Xms32m", "-Xmx384m", "-jar", Path.of(location).toString(), "--parse", sourceFormat, source.toString())
            .redirectOutput(output.toFile()).redirectError(ProcessBuilder.Redirect.DISCARD).start();
        if (!process.waitFor(settings.timeoutSeconds(), TimeUnit.SECONDS)) {
            process.destroyForcibly();
            process.waitFor();
            throw new ParseError("legacy_parse_timeout");
        }
        if (process.exitValue() != 0 || !Files.isRegularFile(output) || Files.size(output) < 2) throw new ParseError("legacy_parse_failed");
        if (Files.size(output) > settings.maxBytes()) throw new ParseError("legacy_parse_output_too_large");
        byte[] payload = Files.readAllBytes(output);
        if (payload[0] != '{') throw new ParseError("legacy_parse_invalid_output");
        return payload;
    }

    private static boolean authorized(String expected, String provided) {
        return expected != null && provided != null && !expected.isEmpty()
            && MessageDigest.isEqual(expected.getBytes(StandardCharsets.UTF_8), provided.getBytes(StandardCharsets.UTF_8));
    }

    private static void respond(HttpExchange exchange, int status, String contentType, byte[] payload) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", contentType);
        exchange.sendResponseHeaders(status, payload.length);
        try (OutputStream output = exchange.getResponseBody()) { output.write(payload); }
    }

    private static void error(HttpExchange exchange, int status, String code) throws IOException {
        exchange.getResponseHeaders().set("X-Parser-Error", code);
        respond(exchange, status, "text/plain", new byte[0]);
    }

    private static void deleteTree(Path root) throws IOException {
        if (!Files.exists(root)) return;
        try (var paths = Files.walk(root)) { paths.sorted((left, right) -> right.compareTo(left)).forEach(path -> { try { Files.deleteIfExists(path); } catch (IOException ignored) {} }); }
    }

    private static void healthcheck() throws Exception {
        URL url = new URL("http://127.0.0.1:" + positiveEnv("PORT", 8090) + "/healthz");
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setConnectTimeout(3000);
        connection.setReadTimeout(3000);
        if (connection.getResponseCode() != 200) throw new IOException("unhealthy");
    }
}
