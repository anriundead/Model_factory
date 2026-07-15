package org.mergenetic.legacyparser;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.apache.poi.hslf.usermodel.HSLFSlide;
import org.apache.poi.hslf.usermodel.HSLFSlideShow;
import org.apache.poi.hslf.usermodel.HSLFTextBox;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LegacyOfficeParserTest {
    @Test
    void legacyFormatsExposeOnlyTruthfulCitationKinds() throws Exception {
        assertEquals("paragraph", LegacyOfficeParser.locatorKindFor("doc"));
        assertEquals("slide", LegacyOfficeParser.locatorKindFor("ppt"));
        assertThrows(Exception.class, () -> LegacyOfficeParser.locatorKindFor("docx"));
        Exception missingFormat = assertThrows(Exception.class, () -> LegacyOfficeParser.locatorKindFor(null));
        assertEquals("unsupported_legacy_format", missingFormat.getMessage());
    }

    @Test
    void parsesLegacyPowerPointIntoSlideLocator(@TempDir Path root) throws Exception {
        Path source = root.resolve("briefing.ppt");
        try (HSLFSlideShow presentation = new HSLFSlideShow()) {
            HSLFSlide slide = presentation.createSlide();
            HSLFTextBox text = new HSLFTextBox();
            text.setText("verified finding");
            slide.addShape(text);
            try (var output = Files.newOutputStream(source)) { presentation.write(output); }
        }

        List<?> sections = LegacyOfficeParser.parseDocument(source, "ppt");
        assertEquals(1, sections.size());
        assertEquals("slide", ((LegacyOfficeParser.Section) sections.get(0)).kind());
        assertEquals(1, ((LegacyOfficeParser.Section) sections.get(0)).value());
        assertEquals("verified finding", ((LegacyOfficeParser.Section) sections.get(0)).text());
    }
}
