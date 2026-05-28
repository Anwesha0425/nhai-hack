const util = require('util');
if (!util.styleText) {
  // Polyfill styleText for Node v21.5.0 so Metro reporter doesn't crash
  util.styleText = (format, text) => text;
}
// Pass execution to Expo CLI
require('expo/bin/cli');
