#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace stemsplitter
{
namespace Colours
{
    const juce::Colour background   { 0xff17191e };
    const juce::Colour panel        { 0xff20232a };
    const juce::Colour panelLight   { 0xff2a2e37 };
    const juce::Colour outline      { 0xff363b46 };
    const juce::Colour text         { 0xffe8eaef };
    const juce::Colour textDim      { 0xff8b93a3 };
    const juce::Colour accent       { 0xffff8a1f };   // FL-ish orange
    const juce::Colour danger       { 0xffff5a5f };

    inline juce::Colour forStem (const juce::String& name)
    {
        if (name == "Vocals")       return juce::Colour (0xffff5c8a);
        if (name == "Drums")        return juce::Colour (0xffffa63d);
        if (name == "Bass")         return juce::Colour (0xff8f7bff);
        if (name == "Other")        return juce::Colour (0xff35c9b3);
        if (name == "Guitar")       return juce::Colour (0xffffd74a);
        if (name == "Piano")        return juce::Colour (0xff4fb0ff);
        if (name == "Instrumental") return juce::Colour (0xff7fd46b);
        return juce::Colour (0xffb0b8c8);
    }
}

class StemLookAndFeel : public juce::LookAndFeel_V4
{
public:
    StemLookAndFeel()
    {
        setColour (juce::ResizableWindow::backgroundColourId, Colours::background);
        setColour (juce::TextButton::buttonColourId, Colours::panelLight);
        setColour (juce::TextButton::buttonOnColourId, Colours::accent);
        setColour (juce::TextButton::textColourOffId, Colours::text);
        setColour (juce::TextButton::textColourOnId, juce::Colours::black);
        setColour (juce::ComboBox::backgroundColourId, Colours::panelLight);
        setColour (juce::ComboBox::outlineColourId, Colours::outline);
        setColour (juce::ComboBox::textColourId, Colours::text);
        setColour (juce::ComboBox::arrowColourId, Colours::textDim);
        setColour (juce::PopupMenu::backgroundColourId, Colours::panel);
        setColour (juce::PopupMenu::highlightedBackgroundColourId, Colours::accent);
        setColour (juce::PopupMenu::highlightedTextColourId, juce::Colours::black);
        setColour (juce::PopupMenu::textColourId, Colours::text);
        setColour (juce::ToggleButton::textColourId, Colours::text);
        setColour (juce::ToggleButton::tickColourId, Colours::accent);
        setColour (juce::ToggleButton::tickDisabledColourId, Colours::textDim);
        setColour (juce::Label::textColourId, Colours::text);
        setColour (juce::ProgressBar::backgroundColourId, Colours::panelLight);
        setColour (juce::ProgressBar::foregroundColourId, Colours::accent);
        setColour (juce::AlertWindow::backgroundColourId, Colours::panel);
        setColour (juce::AlertWindow::textColourId, Colours::text);
        setColour (juce::AlertWindow::outlineColourId, Colours::outline);
    }

    void drawButtonBackground (juce::Graphics& g, juce::Button& button, const juce::Colour& backgroundColour,
                               bool isHighlighted, bool isDown) override
    {
        auto bounds = button.getLocalBounds().toFloat().reduced (0.5f);
        auto colour = button.getToggleState() ? button.findColour (juce::TextButton::buttonOnColourId)
                                              : backgroundColour;

        if (! button.isEnabled())   colour = colour.withMultipliedAlpha (0.4f);
        else if (isDown)            colour = colour.darker (0.2f);
        else if (isHighlighted)     colour = colour.brighter (0.12f);

        g.setColour (colour);
        g.fillRoundedRectangle (bounds, 5.0f);
    }

    juce::Font getTextButtonFont (juce::TextButton&, int buttonHeight) override
    {
        return juce::Font (juce::FontOptions (juce::jmin (15.0f, (float) buttonHeight * 0.5f)));
    }

    juce::Font getComboBoxFont (juce::ComboBox&) override
    {
        return juce::Font (juce::FontOptions (14.0f));
    }

    juce::Font getPopupMenuFont() override
    {
        return juce::Font (juce::FontOptions (14.0f));
    }

    void drawProgressBar (juce::Graphics& g, juce::ProgressBar&, int width, int height,
                          double progress, const juce::String& text) override
    {
        auto bounds = juce::Rectangle<float> (0.0f, 0.0f, (float) width, (float) height);
        g.setColour (Colours::panelLight);
        g.fillRoundedRectangle (bounds, height * 0.5f);

        if (progress > 0.0)
        {
            g.setColour (Colours::accent);
            g.fillRoundedRectangle (bounds.withWidth ((float) (width * juce::jlimit (0.0, 1.0, progress))),
                                    height * 0.5f);
        }

        if (text.isNotEmpty())
        {
            g.setColour (Colours::text);
            g.setFont (juce::FontOptions (12.0f));
            g.drawText (text, bounds, juce::Justification::centred);
        }
    }
};
} // namespace stemsplitter
