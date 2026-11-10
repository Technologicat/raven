---
name: erisian-forecast
description: Deliver the Erisian forecast for a place, and queue the track it calls for. Use when the user asks for the Erisian forecast.
---

Follow these steps in order. Do not skip one because its result seems obvious.

1. **The date.** Take today's date from `get_current_time`. Let `Y` be the year, and let `N` be the day of the month plus the month's offset: January 0, February 31, March 59, April 90, May 120, June 151, July 181, August 212, September 243, October 273, November 304, December 334. Compute `N` with `calculate`, and use that number below as it is, even in a leap year. Use `calculate` for each of these, and do not work them out yourself:
   - season number: `(N - 1) // 73`, which indexes Chaos, Discord, Confusion, Bureaucracy, The Aftermath
   - day of the season: `(N - 1) % 73 + 1`
   - weekday number: `(N - 1) % 5`, which indexes Sweetmorn, Boomtime, Pungenday, Prickle-Prickle, Setting Orange
   - year: `Y + 1166`

2. **The weather.** Geocode the place the user named with `open-meteo__geocoding`, then call `open-meteo__weather_forecast` with `current_weather` set to true, and read its `weathercode`.

3. **The word.** The weather code selects one word:
   - 0 or 1: `clear`
   - 2 or 3: `cloud`
   - 45 to 48: `fog`
   - 51 to 67, or 80 to 82: `rain`
   - 71 to 77, or 85 or 86: `snow`
   - 95 and above: `storm`

4. **The track.** Call `rhythmbox__search_tracks` with the word. If nothing matches, remove the word's last letter and search again, until something matches. From the results take the track with the longest title, the first listed if several tie. Call `rhythmbox__queue_track` with its path.

5. **The forecast.** Reply in exactly three sentences. The first states the date in the form `Weekday, D Season YYYY`. The second relates the weather to the track you queued, naming the track. The third is exactly: `Hail Eris.`
